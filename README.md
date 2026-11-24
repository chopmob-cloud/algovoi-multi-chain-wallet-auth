# algovoi-multi-chain-wallet-auth


[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

Multi-chain wallet authentication primitives, extracted from the AlgoVoi
production gateway in support of
[A2A #1737](https://github.com/a2aproject/A2A/discussions/1737).

A SIWE-inspired challenge-response pattern that lets a customer prove ownership
of a self-custody wallet without an email and password. The challenge message
states explicitly that signing proves ownership only and does not authorise any
payment or recurring charge; the private key never leaves the wallet.

## Supported chains (v1)

- EVM (Base): EIP-191 `personal_sign` over a UTF-8 message
- Algorand / VOI: Ed25519 over the message
- Solana: Ed25519 over the message

Stellar, Hedera, and Tempo are deliberately deferred for v1.

## Usage

Single-module package (`wallet_auth.py`): build a challenge for a wallet
address, then verify the returned signature against that challenge.

## License

Apache 2.0. See [LICENSE](./LICENSE).
