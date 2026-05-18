"""Unit tests for wallet_auth: challenge generation + multi-chain signature
verification + customer JWT issuance / decode."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
import nacl.signing
import pytest

from wallet_auth import (
    chain_to_family,
    decode_customer_jwt,
    generate_challenge,
    issue_customer_jwt,
    verify_signature,
    _b58decode,
)


# ---------------------------------------------------------------------------
# chain_to_family
# ---------------------------------------------------------------------------

class TestChainFamily:
    def test_base_evm(self):
        assert chain_to_family("base_mainnet") == "evm"
        assert chain_to_family("base_sepolia") == "evm"

    def test_eip155_evm(self):
        assert chain_to_family("eip155:8453") == "evm"

    def test_algorand(self):
        assert chain_to_family("algorand_mainnet") == "algorand"
        assert chain_to_family("voi_testnet") == "algorand"

    def test_solana(self):
        assert chain_to_family("solana_mainnet") == "solana"

    def test_tempo_evm(self):
        # Tempo (chain ID 4217) is EVM-compatible — same personal_sign scheme.
        assert chain_to_family("tempo_mainnet") == "evm"
        assert chain_to_family("tempo_testnet") == "evm"

    def test_arc_evm(self):
        # ARC Network (chain ID 5042002) is EVM L1 — same personal_sign scheme.
        assert chain_to_family("arc_testnet") == "evm"

    def test_stellar(self):
        assert chain_to_family("stellar_mainnet") == "stellar"
        assert chain_to_family("stellar_testnet") == "stellar"

    def test_hedera(self):
        assert chain_to_family("hedera_mainnet") == "hedera"
        assert chain_to_family("hedera_testnet") == "hedera"

    def test_unsupported_chain_raises(self):
        for c in ("aptos_mainnet", "near_mainnet", "sui_mainnet", "cosmos"):
            with pytest.raises(ValueError):
                chain_to_family(c)

    def test_canonical_hyphen_form(self):
        """x402 v2 canonical chain names use hyphens (`base-mainnet` etc.),
        not underscores. The normaliser must accept all three forms (bare,
        underscore, hyphen) plus CAIP-2."""
        # Hyphen form — x402 v2 canonical (would have 400'd pre-fix on
        # /v1/recurr/auth/challenge — externally observed 2026-05-16).
        assert chain_to_family("base-mainnet")     == "evm"
        assert chain_to_family("base-sepolia")     == "evm"
        assert chain_to_family("tempo-mainnet")    == "evm"
        assert chain_to_family("arc-testnet")      == "evm"
        assert chain_to_family("algorand-mainnet") == "algorand"
        assert chain_to_family("voi-mainnet")      == "algorand"
        assert chain_to_family("solana-mainnet")   == "solana"
        assert chain_to_family("stellar-mainnet")  == "stellar"
        assert chain_to_family("hedera-mainnet")   == "hedera"

    def test_bare_chain_names(self):
        """The shortest accepted form (no suffix). Used by the live
        /v1/recurr/auth/challenge endpoint and several internal callers."""
        assert chain_to_family("base")     == "evm"
        assert chain_to_family("tempo")    == "evm"
        assert chain_to_family("arc")      == "evm"
        assert chain_to_family("algorand") == "algorand"
        assert chain_to_family("voi")      == "algorand"
        assert chain_to_family("solana")   == "solana"
        assert chain_to_family("stellar")  == "stellar"
        assert chain_to_family("hedera")   == "hedera"


# ---------------------------------------------------------------------------
# generate_challenge
# ---------------------------------------------------------------------------

class TestGenerateChallenge:
    def test_message_includes_required_anchors(self):
        chal = generate_challenge(chain="base_mainnet", wallet_address="0xABC", issuer_domain="recurr.dev")
        assert "AlgoVoi Recurr" in chal.message
        assert "0xABC" in chal.message
        assert "recurr.dev" in chal.message
        assert chal.nonce in chal.message
        # Defensive language — no "approve", "authorise payment", "mandate"
        assert "authorise" not in chal.message.lower() or "does NOT authorise" in chal.message
        assert "It does NOT authorise any payment" in chal.message

    def test_unique_nonce_per_call(self):
        a = generate_challenge(chain="base_mainnet", wallet_address="0xA")
        b = generate_challenge(chain="base_mainnet", wallet_address="0xA")
        assert a.nonce != b.nonce
        assert a.message != b.message

    def test_expires_at_in_future(self):
        now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        chal = generate_challenge(chain="base_mainnet", wallet_address="0xA", ttl_seconds=300, now_utc=now)
        assert chal.expires_at == now + timedelta(seconds=300)

    def test_unsupported_chain_raises(self):
        # Hedera, Stellar, Tempo are all supported now — pick a chain we
        # genuinely don't yet support (Aptos is not on the platform).
        with pytest.raises(ValueError):
            generate_challenge(chain="aptos_mainnet", wallet_address="0xA")


# ---------------------------------------------------------------------------
# verify_signature — EVM
# ---------------------------------------------------------------------------

class TestVerifyEvm:
    @pytest.fixture
    def evm_key(self):
        from eth_account import Account
        Account.enable_unaudited_hdwallet_features()
        acct = Account.create()
        return acct

    def test_valid_signature_accepted(self, evm_key):
        from eth_account.messages import encode_defunct
        msg = "Hello Recurr"
        signed = evm_key.sign_message(encode_defunct(text=msg))
        assert verify_signature(
            chain="base_mainnet",
            wallet_address=evm_key.address,
            message=msg,
            signature=signed.signature.hex(),
        )

    def test_signature_bound_to_address(self, evm_key):
        """Replaying a valid signature against a different address must fail."""
        from eth_account.messages import encode_defunct
        from eth_account import Account

        msg = "Hello Recurr"
        signed = evm_key.sign_message(encode_defunct(text=msg))
        other = Account.create()
        assert not verify_signature(
            chain="base_mainnet",
            wallet_address=other.address,
            message=msg,
            signature=signed.signature.hex(),
        )

    def test_signature_bound_to_message(self, evm_key):
        """Tampering with the message after signing must fail verification."""
        from eth_account.messages import encode_defunct

        signed = evm_key.sign_message(encode_defunct(text="original"))
        assert not verify_signature(
            chain="base_mainnet",
            wallet_address=evm_key.address,
            message="tampered",
            signature=signed.signature.hex(),
        )

    def test_garbage_signature_rejected(self, evm_key):
        assert not verify_signature(
            chain="base_mainnet",
            wallet_address=evm_key.address,
            message="x",
            signature="0x" + "00" * 65,
        )


# ---------------------------------------------------------------------------
# verify_signature — Solana (ed25519)
# ---------------------------------------------------------------------------

class TestVerifySolana:
    def _b58encode(self, data: bytes) -> str:
        ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        n = int.from_bytes(data, "big")
        out = ""
        while n:
            n, r = divmod(n, 58)
            out = ALPHABET[r] + out
        zeros = 0
        for b in data:
            if b == 0:
                zeros += 1
            else:
                break
        return ("1" * zeros) + out

    def test_valid_signature_accepted(self):
        sk = nacl.signing.SigningKey.generate()
        vk = sk.verify_key
        address = self._b58encode(bytes(vk))
        msg = "hello solana"
        sig = sk.sign(msg.encode("utf-8")).signature
        assert verify_signature(
            chain="solana_mainnet",
            wallet_address=address,
            message=msg,
            signature=base64.b64encode(sig).decode(),
        )

    def test_wrong_pubkey_rejected(self):
        sk = nacl.signing.SigningKey.generate()
        sig = sk.sign(b"x").signature
        # different keypair address
        other = nacl.signing.SigningKey.generate()
        addr = self._b58encode(bytes(other.verify_key))
        assert not verify_signature(
            chain="solana_mainnet",
            wallet_address=addr,
            message="x",
            signature=base64.b64encode(sig).decode(),
        )

    def test_b58decode_roundtrip(self):
        for original in [b"\x01\x02\x03", b"\x00\x00\xff", bytes(range(32))]:
            encoded = self._b58encode(original)
            assert _b58decode(encoded) == original


# ---------------------------------------------------------------------------
# verify_signature — Algorand (ed25519)
# ---------------------------------------------------------------------------

class TestVerifyAlgorand:
    def test_valid_signature_accepted(self):
        try:
            from algosdk.account import generate_account
            from algosdk.encoding import decode_address
        except ImportError:
            pytest.skip("algosdk not installed")
        # algosdk's generate_account returns (private_key_b64, address)
        # The private key is 32-byte seed + 32-byte pubkey concatenated, base64'd
        priv_b64, address = generate_account()
        seed = base64.b64decode(priv_b64)[:32]
        sk = nacl.signing.SigningKey(seed)
        msg = "hello algorand"
        sig = sk.sign(msg.encode()).signature
        assert verify_signature(
            chain="algorand_mainnet",
            wallet_address=address,
            message=msg,
            signature=base64.b64encode(sig).decode(),
        )

    def test_wrong_address_rejected(self):
        try:
            from algosdk.account import generate_account
        except ImportError:
            pytest.skip("algosdk not installed")
        priv_b64, _addr = generate_account()
        seed = base64.b64decode(priv_b64)[:32]
        sk = nacl.signing.SigningKey(seed)
        msg = "x"
        sig = sk.sign(msg.encode()).signature
        # use a different account's address
        _, other_addr = generate_account()
        assert not verify_signature(
            chain="algorand_mainnet",
            wallet_address=other_addr,
            message=msg,
            signature=base64.b64encode(sig).decode(),
        )

    def test_mx_prefixed_signature_accepted(self):
        """Pera Wallet, Defly, Lute, Kibisis et al. prepend 'MX' to the
        message bytes before signing (AlgoSigner-compatible signData
        convention). The verifier must accept both raw and MX-prefixed
        framings without leaking which one matched."""
        try:
            from algosdk.account import generate_account
        except ImportError:
            pytest.skip("algosdk not installed")
        priv_b64, address = generate_account()
        seed = base64.b64decode(priv_b64)[:32]
        sk = nacl.signing.SigningKey(seed)
        msg = "AlgoVoi Recurr challenge\nNonce: abc123"
        # Pera/Defly/Lute style: sign b"MX" || utf8(message)
        sig = sk.sign(b"MX" + msg.encode()).signature
        assert verify_signature(
            chain="algorand_mainnet",
            wallet_address=address,
            message=msg,
            signature=base64.b64encode(sig).decode(),
        )
        # Same wallet, same key, VOI chain — should also pass since the
        # family resolves identically.
        assert verify_signature(
            chain="voi_mainnet",
            wallet_address=address,
            message=msg,
            signature=base64.b64encode(sig).decode(),
        )

    def test_mxmx_double_prefixed_signature_accepted(self):
        """The AlgoVoi Chrome extension v1.0.x manually prepends 'MX' before
        calling algosdk.signBytes — but algosdk 3.x already prepends 'MX'
        per ARC-1, producing a double-MX framing. The verifier must accept
        this so existing extension installs work."""
        try:
            from algosdk.account import generate_account
        except ImportError:
            pytest.skip("algosdk not installed")
        priv_b64, address = generate_account()
        seed = base64.b64decode(priv_b64)[:32]
        sk = nacl.signing.SigningKey(seed)
        msg = "AlgoVoi Recurr challenge\nNonce: algovoi-ext-bug"
        sig = sk.sign(b"MXMX" + msg.encode()).signature
        assert verify_signature(
            chain="algorand_mainnet",
            wallet_address=address,
            message=msg,
            signature=base64.b64encode(sig).decode(),
        )

    def test_mx_prefixed_signature_with_wrong_address_rejected(self):
        """Replaying an MX-prefixed signature against a different address
        must fail (binding to address must hold for both framings)."""
        try:
            from algosdk.account import generate_account
        except ImportError:
            pytest.skip("algosdk not installed")
        priv_b64, _addr = generate_account()
        seed = base64.b64decode(priv_b64)[:32]
        sk = nacl.signing.SigningKey(seed)
        msg = "x"
        sig = sk.sign(b"MX" + msg.encode()).signature
        _, other_addr = generate_account()
        assert not verify_signature(
            chain="algorand_mainnet",
            wallet_address=other_addr,
            message=msg,
            signature=base64.b64encode(sig).decode(),
        )


# ---------------------------------------------------------------------------
# verify_signature — Stellar (Freighter, Stellar Wallets Kit)
# ---------------------------------------------------------------------------


class TestVerifyStellar:
    def test_valid_signature_accepted(self):
        try:
            from stellar_sdk import Keypair
        except ImportError:
            pytest.skip("stellar_sdk not installed")
        kp = Keypair.random()
        msg = "AlgoVoi Recurr challenge\nNonce: stellar123"
        sig = kp.sign(msg.encode("utf-8"))
        assert verify_signature(
            chain="stellar_mainnet",
            wallet_address=kp.public_key,
            message=msg,
            signature=base64.b64encode(sig).decode(),
        )

    def test_wrong_address_rejected(self):
        try:
            from stellar_sdk import Keypair
        except ImportError:
            pytest.skip("stellar_sdk not installed")
        kp = Keypair.random()
        other = Keypair.random()
        msg = "x"
        sig = kp.sign(msg.encode())
        assert not verify_signature(
            chain="stellar_mainnet",
            wallet_address=other.public_key,
            message=msg,
            signature=base64.b64encode(sig).decode(),
        )

    def test_tampered_message_rejected(self):
        try:
            from stellar_sdk import Keypair
        except ImportError:
            pytest.skip("stellar_sdk not installed")
        kp = Keypair.random()
        sig = kp.sign(b"original")
        assert not verify_signature(
            chain="stellar_mainnet",
            wallet_address=kp.public_key,
            message="tampered",
            signature=base64.b64encode(sig).decode(),
        )

    def test_malformed_address_rejected(self):
        # Must NOT raise — fail-closed.
        assert not verify_signature(
            chain="stellar_mainnet",
            wallet_address="not-a-stellar-address",
            message="x",
            signature=base64.b64encode(b"\x00" * 64).decode(),
        )


# ---------------------------------------------------------------------------
# verify_signature — Hedera (mirror-node-resolved pubkey)
# ---------------------------------------------------------------------------


class TestVerifyHedera:
    def test_ed25519_raw_signature_accepted(self):
        """ED25519 Hedera account, signature over raw UTF-8 message bytes."""
        sk = nacl.signing.SigningKey.generate()
        pubkey_der = "302a300506032b6570032100" + bytes(sk.verify_key).hex()
        msg = "hedera challenge"
        sig = sk.sign(msg.encode()).signature
        assert verify_signature(
            chain="hedera_mainnet",
            wallet_address="0.0.1234",
            message=msg,
            signature=base64.b64encode(sig).decode(),
            pubkey=pubkey_der,
        )

    def test_ed25519_eip191_envelope_accepted(self):
        """Hedera Wallet Connect 2.x prefixes with '\\x19Hedera Signed
        Message:\\n<len>' before signing — the verifier must accept it."""
        sk = nacl.signing.SigningKey.generate()
        pubkey_der = "302a300506032b6570032100" + bytes(sk.verify_key).hex()
        msg = "envelope test"
        envelope = b"\x19Hedera Signed Message:\n" + str(len(msg)).encode() + msg.encode()
        sig = sk.sign(envelope).signature
        assert verify_signature(
            chain="hedera_mainnet",
            wallet_address="0.0.1234",
            message=msg,
            signature=base64.b64encode(sig).decode(),
            pubkey=pubkey_der,
        )

    def test_ed25519_wrong_pubkey_rejected(self):
        sk = nacl.signing.SigningKey.generate()
        other_sk = nacl.signing.SigningKey.generate()
        pubkey_der = "302a300506032b6570032100" + bytes(other_sk.verify_key).hex()
        msg = "x"
        sig = sk.sign(msg.encode()).signature
        assert not verify_signature(
            chain="hedera_mainnet",
            wallet_address="0.0.1234",
            message=msg,
            signature=base64.b64encode(sig).decode(),
            pubkey=pubkey_der,
        )

    def test_ecdsa_secp256k1_signature_accepted(self):
        """ECDSA secp256k1 Hedera account (EVM-compatible). Signature is
        over keccak256(EIP-191 envelope) per HIP-820."""
        try:
            from eth_keys import keys
            from eth_utils.crypto import keccak
        except ImportError:
            pytest.skip("eth_keys / eth_utils not installed")
        priv = keys.PrivateKey(b"\x42" * 32)
        compressed = priv.public_key.to_compressed_bytes()
        pubkey_der = "302d300706052b8104000a032200" + compressed.hex()

        msg = "ecdsa hedera challenge"
        envelope = b"\x19Hedera Signed Message:\n" + str(len(msg)).encode() + msg.encode()
        digest = keccak(envelope)
        sig = priv.sign_msg_hash(digest)
        assert verify_signature(
            chain="hedera_mainnet",
            wallet_address="0.0.5678",
            message=msg,
            signature=base64.b64encode(sig.to_bytes()).decode(),
            pubkey=pubkey_der,
        )

    def test_missing_pubkey_rejected(self):
        """Hedera verification must fail-closed when caller forgot to
        supply the pubkey (i.e. the mirror-node lookup failed upstream)."""
        sk = nacl.signing.SigningKey.generate()
        msg = "x"
        sig = sk.sign(msg.encode()).signature
        assert not verify_signature(
            chain="hedera_mainnet",
            wallet_address="0.0.1234",
            message=msg,
            signature=base64.b64encode(sig).decode(),
            # pubkey omitted on purpose
        )


# ---------------------------------------------------------------------------
# Customer JWT
# ---------------------------------------------------------------------------

class TestCustomerJwt:
    """JWT is now wallet-first: no tenant or customer_id claim. The wallet IS
    the identity; cross-tenant resolution happens at request time."""
    SECRET = "test-secret-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    def test_issue_and_decode_roundtrip(self):
        token, exp = issue_customer_jwt(
            wallet_address="0xABC",
            chain="base_mainnet",
            secret=self.SECRET,
        )
        session = decode_customer_jwt(token, secret=self.SECRET)
        assert session.wallet_address == "0xABC"
        assert session.chain == "base_mainnet"
        assert session.expires_at == int(exp.timestamp())

    def test_wrong_secret_rejected(self):
        token, _ = issue_customer_jwt(
            wallet_address="0xA", chain="base_mainnet", secret=self.SECRET,
        )
        with pytest.raises(pyjwt.PyJWTError):
            decode_customer_jwt(token, secret="wrong-secret")

    def test_expired_jwt_rejected(self):
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        token, _ = issue_customer_jwt(
            wallet_address="0xA", chain="base_mainnet", secret=self.SECRET,
            ttl_seconds=60, now_utc=past,
        )
        with pytest.raises(pyjwt.ExpiredSignatureError):
            decode_customer_jwt(token, secret=self.SECRET)

    def test_wrong_scope_rejected(self):
        # A token with the right secret but a different scope must NOT decode
        # as a customer session — protects against tenant API tokens being
        # abused as customer JWTs.
        payload = {
            "sub": "0xA", "wal": "0xA", "chn": "base_mainnet",
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "exp": int((datetime.now(timezone.utc) + timedelta(minutes=5)).timestamp()),
            "scope": "tenant.admin",
        }
        token = pyjwt.encode(payload, self.SECRET, algorithm="HS256")
        with pytest.raises(pyjwt.InvalidTokenError):
            decode_customer_jwt(token, secret=self.SECRET)
