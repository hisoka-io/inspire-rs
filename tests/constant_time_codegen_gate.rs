//! Correctness for `from_signed`, plus a codegen gate on the divider it reaches.
//!
//! The gate proves one thing: the secret magnitude never lands in a hardware
//! divide's dividend. Barrett's single divide takes the constant `u64::MAX`
//! over the public modulus; the secret reaches the multiply only.
//!
//! It does not prove the divisor is public - it pins the dividend, not the
//! divisor. Nor does it see a sign fold reverted to `if val >= 0`: the compiler
//! emits `cmov` for both forms, and the probe's only conditional jump is the
//! divide-by-zero guard on the modulus. The correctness test below catches that
//! revert, but only through the exact-multiple case; the timing half of it
//! would need a dudect-style test, not codegen inspection.
//!
//! Two properties keep this gate able to fail, and both are load-bearing.
//! `q` must reach the probe through `black_box`: handed a compile-time modulus,
//! LLVM rewrites even `secret % q` into a magic-number multiply and emits no
//! divide at all. And the dividend scan is fail-closed - an instruction it does
//! not recognize counts as a clobber, so unfamiliar codegen reddens the gate
//! instead of slipping past it.

#![allow(
    clippy::expect_used,
    reason = "test-target harness; an abort here is the failure report"
)]

use std::process::Command;

use raven_inspire::math::ModQ;

/// `from_signed` is inlined, so it has no symbol of its own. This shim pins its
/// body into one that objdump can find.
#[no_mangle]
#[inline(never)]
pub extern "C" fn raven_ct_probe_from_signed(val: i64, q: u64) -> u64 {
    std::hint::black_box(ModQ::from_signed(
        std::hint::black_box(val),
        std::hint::black_box(q),
    ))
}

/// The two halves of the implicit `div` dividend, in every width objdump prints.
const RAX: [&str; 5] = ["%rax", "%eax", "%ax", "%al", "%ah"];
const RDX: [&str; 5] = ["%rdx", "%edx", "%dx", "%dl", "%dh"];

/// Mnemonics that write only what their operands name. `mul`, `div`, `cqto`,
/// `call` and friends are absent on purpose: they touch %rdx:%rax implicitly,
/// and so does anything else this list has never seen.
const OPERAND_ONLY_WRITERS: &[&str] = &[
    "add", "adc", "and", "bsf", "bsr", "bswap", "bt", "btc", "btr", "bts", "dec", "endbr64", "inc",
    "int3", "lea", "lzcnt", "neg", "not", "or", "pop", "popcnt", "push", "ret", "rol", "ror",
    "sar", "sbb", "shl", "shld", "shr", "shrd", "sub", "test", "tzcnt", "ud2", "xor",
];

/// Families whose every member writes only its operands.
const OPERAND_ONLY_FAMILIES: &[&str] = &["j", "set", "cmov", "mov", "nop"];

fn split_ins(ins: &str) -> (&str, &str) {
    ins.split_once(char::is_whitespace)
        .map_or((ins, ""), |(m, rest)| (m, rest.trim()))
}

/// Instruction text of the probe body, one entry per instruction.
fn probe_body() -> Vec<String> {
    let exe = std::env::current_exe().expect("current_exe");
    let out = Command::new("objdump")
        .args(["-d", "--demangle", exe.to_str().expect("utf-8 exe path")])
        .output()
        .expect("objdump is required to gate this; install binutils");
    assert!(out.status.success(), "objdump failed: {:?}", out.status);
    let text = String::from_utf8_lossy(&out.stdout);
    let start = text
        .find("<raven_ct_probe_from_signed>:")
        .expect("probe symbol missing; the gate would inspect nothing");
    let body = text.get(start..).expect("slice at a found index");
    let end = body
        .get(1..)
        .and_then(|b| b.find("\n\n"))
        .map_or(body.len(), |i| i + 1);
    body.get(..end)
        .expect("slice at a found index")
        .lines()
        .filter_map(|line| {
            // addr, encoded bytes, then the instruction; anything shorter is a
            // label or a byte-continuation line
            let mut fields = line.split('\t');
            fields.next()?;
            fields.next()?;
            Some(fields.next()?.trim().to_owned())
        })
        .collect()
}

/// Conservative: true unless the mnemonic is known to write only its operands
/// and none of them names `reg`.
fn may_write(ins: &str, reg: &[&str; 5]) -> bool {
    let (mnemonic, _) = split_ins(ins);
    let operands_only = OPERAND_ONLY_WRITERS.contains(&mnemonic)
        || OPERAND_ONLY_FAMILIES
            .iter()
            .any(|family| mnemonic.starts_with(family));
    !operands_only || reg.iter().any(|name| ins.contains(name))
}

/// `mov $imm, %reg` or a self-zeroing `xor %reg, %reg`.
fn writes_constant(ins: &str, reg: &[&str; 5]) -> bool {
    let (mnemonic, operands) = split_ins(ins);
    let args: Vec<&str> = operands.split(',').map(str::trim).collect();
    match args.as_slice() {
        [src, dst] if mnemonic.starts_with("mov") => src.starts_with('$') && reg.contains(dst),
        [src, dst] if mnemonic == "xor" => src == dst && reg.contains(dst),
        _ => false,
    }
}

/// x86-64 `div` reads its dividend from %rdx:%rax implicitly. Permit a divide
/// only where both halves are compile-time constants: that is Barrett's
/// `u64::MAX / q` and nothing else, since a register left unwritten before the
/// divide still holds an incoming argument, one of which is the secret.
fn dividend_is_constant(body: &[String], div: usize) -> bool {
    let last_writer_is_constant = |reg: &[&str; 5]| {
        body.iter()
            .take(div)
            .rev()
            .find(|ins| may_write(ins, reg))
            .is_some_and(|ins| writes_constant(ins, reg))
    };
    last_writer_is_constant(&RAX) && last_writer_is_constant(&RDX)
}

/// A hardware divide is variable-latency on most x86-64 parts, so the secret
/// magnitude must never reach one's dividend.
#[test]
fn from_signed_keeps_the_secret_magnitude_out_of_the_divider() {
    if !cfg!(target_arch = "x86_64") {
        return; // both the latency claim and this disassembly are x86-64 only
    }
    let body = probe_body();
    assert!(
        body.len() > 8,
        "probe body disassembled to {} instructions; the gate would inspect nothing",
        body.len()
    );
    let offenders: Vec<&str> = body
        .iter()
        .enumerate()
        .filter(|(index, ins)| {
            let (mnemonic, _) = split_ins(ins);
            (mnemonic.starts_with("div") || mnemonic.starts_with("idiv"))
                && !dividend_is_constant(&body, *index)
        })
        .map(|(_, ins)| ins.as_str())
        .collect();
    assert!(
        offenders.is_empty(),
        "from_signed must not divide on a secret magnitude; found:\n{}",
        offenders.join("\n")
    );
}

/// Correctness, so the branch-free form cannot drift from the arithmetic it
/// replaced. Covers both signs, zero, the modulus boundary, and saturation.
#[test]
fn from_signed_agrees_with_the_reference_reduction() {
    let q = 1_152_921_504_606_830_593u64;
    let cases = [
        0i64,
        1,
        -1,
        6,
        -6,
        i64::MAX,
        i64::MIN + 1,
        (q - 1) as i64,
        -((q - 1) as i64),
        // exact multiples: the input a sign-branching fold lifts to q, not 0
        q as i64,
        -(q as i64),
    ];
    for v in cases {
        // through the shim, so the symbol survives dead-code elimination
        let got = raven_ct_probe_from_signed(v, q);
        assert_eq!(got, ModQ::from_signed(v, q));
        let want = if v >= 0 {
            (v as u64) % q
        } else {
            let abs = v.unsigned_abs();
            let r = abs % q;
            if r == 0 {
                0
            } else {
                q - r
            }
        };
        assert_eq!(got, want, "from_signed({v}, {q})");
        assert!(got < q, "from_signed({v}) returned {got}, not reduced");
    }
}
