# Security Notes for inspire

- **Supported variants**: InsPIRe^0 (NoPacking), InsPIRe^1 (OnePacking), InsPIRe^2 (Seeded+Packed). These are the only production-supported modes.
- **Modulus-switched responses**: RIMS v2 is exposed behind the default-off `mod-switch-response` feature. Its checked 45-bit target passes the conservative noise gate and byte-identity corpus, but no adapter transport calls it.
- **Rejected target**: The exported 33-bit target fails the conservative production-cell noise gate. Do not wire or enable it.
- **Packed a-side caveat**: At d=256, gamma=4, bounded dense and structured recovery attempts against independently generated actual responses failed. Failure is not a privacy proof; production d=2048 and cross-key security bounds remain unmeasured.
- **Reporting issues**: Please open a GitHub issue in this repository with a minimal reproduction or description. Avoid including sensitive data in issue text.
