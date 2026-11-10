# Security policy

## Supported versions

Security fixes go into the latest released version of pycordis.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately
through GitHub's
[private vulnerability reporting](https://github.com/aswin-kevin/pycordis/security/advisories/new)
for this repository.

Include the pycordis and Python versions, what an attacker can do, and the
smallest steps or config that reproduce it. You should get a reply within 7
days. Once a fix is released, the advisory is published with credit to you
unless you ask otherwise.

## Scope

pycordis has no runtime dependencies. Note the trust model described in the
**Security notes** of [docs/reference.md](docs/reference.md): config files can import and run code (`allow` is
not a sandbox), and the health server is unauthenticated and meant for
localhost or a trusted network. Reports that a trusted config file can run
code are expected behaviour, not vulnerabilities.
