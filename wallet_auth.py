"""
Multi-chain wallet authentication primitives for the Recurr customer portal.

Implements the challenge-response pattern that lets a customer prove ownership
of a self-custody wallet without an email + password. Three chain families are
supported in v1:

  - EVM (Base):     EIP-191 ``personal_sign`` over a UTF-8 message.
                    Signature is 65 bytes hex; address is 20 bytes hex (0x...).
  - Algorand / VOI: ed25519 over the message. Address is base32 with 4-byte
                    checksum (58 chars).
  - Solana:         ed25519 over the message. Address is base58 (32-44 chars).

Stellar / Hedera / Tempo support is deliberately deferred — wallet-connect
ecosystems for those chains are less mature and add disproportionate complexity
for v1.

The challenge message follows the SIWE-inspired template:

  AlgoVoi Recurr wants you to authenticate with your <chain> wallet:
  <wallet_address>

  Signing this message proves ownership. It does NOT authorise any payment
  or recurring charge. AlgoVoi never holds your private key.

  Domain:    <issuer>
  Nonce:     <nonce>
  Issued at: <iso_utc>
  Expires:   <iso_utc>

The wording is deliberately defensive (technical-service-provider framing):
no "approve", "initiate", "mandate", or "authorise payment".
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import jwt

logger = logging.getLogger(__name__)

ChainFamily = Literal["evm", "algorand", "solana", "stellar", "hedera"]
DEFAULT_CHALLENGE_TTL_SECONDS = 300         # 5 minutes
DEFAULT_CUSTOMER_JWT_TTL_SECONDS = 1800     # 30 minutes


# ---------------------------------------------------------------------------
# Chain mapping
# ---------------------------------------------------------------------------


def chain_to_family(chain: str) -> ChainFamily:
    """Map a platform chain string to a wallet-auth family.

    Accepted forms (all normalise to the chain "head"):
      - bare:            ``base``, ``algorand``
      - underscore:      ``base_mainnet``, ``algorand_mainnet`` (platform legacy)
      - hyphen:          ``base-mainnet``, ``algorand-mainnet`` (x402 v2 canonical)
      - CAIP-2:          ``eip155:8453``, ``algorand:mainnet-v1.0``

    Raises ValueError for unsupported chains — callers must surface a 400 to
    the client rather than swallow.
    """
    # Normalise: split on _ (legacy), - (x402 v2 canonical), : (CAIP-2).
    c = chain.lower().split("_", 1)[0].split("-", 1)[0].split(":", 1)[0]
    if c in {"base", "eth", "ethereum", "eip155", "tempo", "arc"}:
        return "evm"
    if c in {"algorand", "voi"}:
        return "algorand"
    if c == "solana":
        return "solana"
    if c == "stellar":
        return "stellar"
    if c == "hedera":
        return "hedera"
    raise ValueError(f"chain '{chain}' is not yet supported for wallet auth")


# ---------------------------------------------------------------------------
# Challenge generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Challenge:
    nonce: str
    message: str
    expires_at: datetime


def generate_challenge(
    *,
    chain: str,
    wallet_address: str,
    issuer_domain: str = "algovoi.co.uk",
    ttl_seconds: int = DEFAULT_CHALLENGE_TTL_SECONDS,
    now_utc: datetime | None = None,
) -> Challenge:
    """Build a single-use challenge for a wallet to sign.

    The message includes nonce, domain, and issued/expires timestamps so the
    customer can verify what they're signing. The text is deliberately
    non-financial — "authenticate" not "authorise", and a parenthetical
    disclaimer that this signature does NOT authorise any payment.
    """
    chain_to_family(chain)  # raises on unsupported chain
    now = now_utc or datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=ttl_seconds)
    nonce = secrets.token_urlsafe(24)

    # Friendly chain label for the signed message — what the wallet popup
    # will show the user. Algorand and VOI share a family for verification
    # but a customer reading "VOI wallet" wants to see "VOI", not "algorand".
    # Accepts all forms: bare, underscore, hyphen (x402 v2 canonical), CAIP-2.
    head = chain.lower().split("_", 1)[0].split("-", 1)[0].split(":", 1)[0]
    chain_label = {
        "algorand": "Algorand",
        "voi": "VOI",
        "solana": "Solana",
        "base": "Base",
        "eth": "Ethereum",
        "ethereum": "Ethereum",
        "eip155": "EVM",
        "tempo": "Tempo",
        "arc": "ARC",
        "stellar": "Stellar",
        "hedera": "Hedera",
    }.get(head, head.capitalize())

    message = (
        f"AlgoVoi Recurr wants you to authenticate with your {chain_label} wallet:\n"
        f"{wallet_address}\n"
        f"\n"
        f"Signing this message proves ownership. It does NOT authorise any payment\n"
        f"or recurring charge. AlgoVoi never holds your private key.\n"
        f"\n"
        f"Domain:    {issuer_domain}\n"
        f"Nonce:     {nonce}\n"
        f"Issued at: {now.replace(microsecond=0).isoformat()}\n"
        f"Expires:   {expires_at.replace(microsecond=0).isoformat()}\n"
    )
    return Challenge(nonce=nonce, message=message, expires_at=expires_at)


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------


def verify_signature(
    *,
    chain: str,
    wallet_address: str,
    message: str,
    signature: str,
    pubkey: str | None = None,
) -> bool:
    """Verify that `signature` was produced by the holder of `wallet_address`
    over `message`. Returns True on success, False otherwise. Never raises.

    Supported families:
      - evm:      EIP-191 personal_sign hex signature (0x-prefixed, 130 hex chars)
                  Includes Tempo (chain ID 4217) — same scheme.
      - algorand: base64-encoded ed25519 signature, raw or "MX"-prefixed framing
      - solana:   base64-encoded ed25519 signature
      - stellar:  base64-encoded ed25519 signature, pubkey derived from G... StrKey
      - hedera:   base64-encoded signature; caller MUST supply ``pubkey`` (DER hex
                  from mirror node). Supports both ED25519 and ECDSA secp256k1
                  account key types — auto-detected from the DER prefix.
    """
    try:
        family = chain_to_family(chain)
    except ValueError:
        return False

    try:
        if family == "evm":
            return _verify_evm(wallet_address, message, signature)
        if family == "algorand":
            return _verify_algorand(wallet_address, message, signature)
        if family == "solana":
            return _verify_solana(wallet_address, message, signature)
        if family == "stellar":
            return _verify_stellar(wallet_address, message, signature)
        if family == "hedera":
            if not pubkey:
                logger.warning(
                    "wallet_auth.hedera_missing_pubkey",
                    extra={"chain": chain, "wallet": wallet_address},
                )
                return False
            return _verify_hedera(wallet_address, message, signature, pubkey)
    except Exception as exc:  # noqa: BLE001 — must be fail-closed on any error
        logger.warning(
            "wallet_auth.verify_failed",
            extra={"chain": chain, "family": family, "error": str(exc)},
        )
        return False
    return False


def _verify_evm(address: str, message: str, signature: str) -> bool:
    from eth_account.messages import encode_defunct
    from eth_account import Account

    if not signature.startswith("0x"):
        signature = "0x" + signature
    if not address.startswith("0x") or len(address) != 42:
        return False

    msg = encode_defunct(text=message)
    recovered = Account.recover_message(msg, signature=signature)
    return recovered.lower() == address.lower()


def _verify_algorand(address: str, message: str, signature: str) -> bool:
    """Algorand / VOI wallets sign with ed25519. The public key is derived by
    base32-decoding the address (and dropping the 4-byte checksum suffix).

    Wallets disagree on framing for arbitrary-data signing:

      - **Raw ed25519** over the UTF-8 message bytes — used by low-level
        ``nacl.SigningKey.sign`` paths (and our unit tests).
      - **"MX" prefix** — the AlgoSigner-compatible signData convention used
        by Pera Wallet's ``signData``, Defly Wallet's ``signData``, Lute's
        ``signData`` and most Algorand browser extensions when signing
        arbitrary bytes (this prevents wallets from being tricked into
        signing a real txn-shaped payload).
      - **"MXMX" prefix** — the AlgoVoi Chrome extension up to v1.0.2
        accidentally double-prefixes: it manually prepends "MX" before
        passing the bytes to ``algosdk.signBytes``, but algosdk 3.x already
        adds the "MX" itself (per ARC-1 domain separation). This produces
        ``ed25519_sign("MX" + "MX" + message)``. Accepting it here lets
        existing extension installs work; the extension will be patched in
        a future release to drop the manual prefix.

    We try every framing and return True if any verifies. Cost of the extra
    ed25519 verifies is sub-millisecond combined and the function is
    fail-closed — neither path leaks why a verification failed.
    """
    import base64
    import nacl.signing
    import nacl.exceptions

    try:
        from algosdk.encoding import decode_address
    except ImportError:
        # Fallback: do the base32 decode + checksum strip manually.
        decode_address = None  # type: ignore[assignment]

    if decode_address is not None:
        public_key_bytes = decode_address(address)
    else:
        import base64 as b64

        # algorand addresses are base32 (no padding) of (pubkey || checksum[4]).
        padded = address + ("=" * ((8 - len(address) % 8) % 8))
        decoded = b64.b32decode(padded)
        public_key_bytes = decoded[:32]

    try:
        sig_bytes = base64.b64decode(signature)
    except Exception:
        return False

    # Pera Wallet over WC v2 only implements `algo_signTxn`, not `algo_signData`.
    # The recurr portal therefore embeds the challenge in a zero-ALGO self-
    # payment transaction's `note` field and submits the signed transaction
    # blob in place of a raw signature.  If the blob looks like a signed
    # Algorand transaction (msgpack map with `sig`+`txn` keys), verify via the
    # transaction path; otherwise fall back to the raw-signature path used by
    # AlgoSigner/Defly/Lute's signData (and our unit tests).
    if _is_signed_algorand_txn(sig_bytes):
        return _verify_algorand_signed_txn(address, message, sig_bytes, public_key_bytes)

    verify_key = nacl.signing.VerifyKey(public_key_bytes)
    msg_bytes = message.encode("utf-8")
    for prefix in (b"", b"MX", b"MXMX"):
        try:
            verify_key.verify(prefix + msg_bytes, sig_bytes)
            return True
        except nacl.exceptions.BadSignatureError:
            continue
    return False


def _is_signed_algorand_txn(blob: bytes) -> bool:
    """Cheap heuristic to distinguish a signed Algorand transaction (msgpack
    map with `sig` and `txn` top-level keys) from a raw ed25519 signature
    (exactly 64 bytes).  Signed txns are always >100 bytes and contain the
    msgpack-encoded fixstr markers `\\xa3sig` and `\\xa3txn` near the start.
    """
    if len(blob) < 80:
        return False
    head = blob[:512]
    return b"\xa3sig" in head and b"\xa3txn" in head


def _verify_algorand_signed_txn(
    address: str,
    expected_message: str,
    blob: bytes,
    public_key_bytes: bytes,
) -> bool:
    """Verify a signed Algorand transaction as proof of address ownership.

    The portal builds a zero-ALGO self-payment with the auth challenge in the
    `note` field, sends it through `algo_signTxn`, then forwards the signed
    blob to this verifier.  We never submit the transaction — we just check:

      1. The msgpack-decoded SignedTransaction has a `note` matching the
         expected challenge bytes.
      2. The sender matches the claimed wallet address.
      3. The ed25519 signature is valid over the canonical txn-to-sign bytes
         (`b"TX" + msgpack(txn)`).
    """
    import nacl.signing
    import nacl.exceptions

    try:
        from algosdk.encoding import msgpack_decode
    except ImportError:
        return False

    try:
        signed = msgpack_decode(blob)
    except Exception:
        return False

    sig = getattr(signed, "signature", None)
    txn = getattr(signed, "transaction", None)
    if not sig or txn is None:
        return False

    note = getattr(txn, "note", None)
    if not note:
        return False
    note_bytes = bytes(note) if not isinstance(note, (bytes, bytearray)) else bytes(note)
    if note_bytes != expected_message.encode("utf-8"):
        return False

    sender = getattr(txn, "sender", None)
    if sender is None or str(sender) != address:
        return False

    try:
        to_be_signed = txn.bytes_to_sign()
    except Exception:
        return False

    try:
        sig_bytes = bytes(sig)
    except Exception:
        return False

    verify_key = nacl.signing.VerifyKey(public_key_bytes)
    try:
        verify_key.verify(to_be_signed, sig_bytes)
        return True
    except nacl.exceptions.BadSignatureError:
        return False


def _verify_solana(address: str, message: str, signature: str) -> bool:
    """Solana wallets sign the raw bytes of the message with the private key
    corresponding to the base58-encoded public key (the wallet address)."""
    import base64
    import nacl.signing
    import nacl.exceptions

    public_key_bytes = _b58decode(address)
    if len(public_key_bytes) != 32:
        return False

    try:
        sig_bytes = base64.b64decode(signature)
    except Exception:
        return False

    verify_key = nacl.signing.VerifyKey(public_key_bytes)
    try:
        verify_key.verify(message.encode("utf-8"), sig_bytes)
        return True
    except nacl.exceptions.BadSignatureError:
        return False


# Inline base58 decoder — avoids adding a new pypi dep just for Solana addresses.
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {c: i for i, c in enumerate(_B58_ALPHABET)}


def _b58decode(s: str) -> bytes:
    n = 0
    for c in s:
        if c not in _B58_INDEX:
            raise ValueError(f"non-base58 char: {c!r}")
        n = n * 58 + _B58_INDEX[c]
    # Length: count leading '1's = leading zero-bytes
    nbytes = (n.bit_length() + 7) // 8
    body = n.to_bytes(nbytes, "big") if n else b""
    leading_ones = len(s) - len(s.lstrip("1"))
    return b"\x00" * leading_ones + body


# ---------------------------------------------------------------------------
# Stellar — StrKey-encoded ed25519 pubkey
# ---------------------------------------------------------------------------


def _strkey_decode_account(address: str) -> bytes:
    """Return the raw 32-byte ed25519 public key from a Stellar G... address.

    StrKey layout: base32(no-pad) of [version_byte=0x30 | pubkey_32 | crc16_le_2].
    We rely on length + version byte; CRC is intentionally not re-validated
    (a valid signature with a wrong CRC is impossible — ed25519 verify covers it).
    """
    import base64 as b64

    if not address or len(address) != 56 or not address.startswith("G"):
        raise ValueError("not a Stellar account address (must be 56-char G...)")
    # base32 must be a multiple of 8 chars; G... addresses already are.
    decoded = b64.b32decode(address)
    if len(decoded) != 35:
        raise ValueError(f"invalid Stellar StrKey length: {len(decoded)}")
    if decoded[0] != 0x30:  # 6 << 3 — account ID version byte
        raise ValueError(f"not an account address (version 0x{decoded[0]:02x})")
    return decoded[1:33]


def _verify_stellar(address: str, message: str, signature: str) -> bool:
    """Stellar wallets (Freighter, Albedo, xBull, Lobstr, Hana, Rabet) sign
    arbitrary messages with raw ed25519 over the UTF-8 message bytes.

    Stellar Wallets Kit's ``verifyStellarSignature`` does the same:
    ``Keypair.verify(messageBytes, signatureBytes)`` — no SEP-10/53 wrapping
    for arbitrary messages. We accept a base64-encoded signature.
    """
    import base64
    import nacl.signing
    import nacl.exceptions

    public_key_bytes = _strkey_decode_account(address)
    try:
        sig_bytes = base64.b64decode(signature)
    except Exception:
        return False

    verify_key = nacl.signing.VerifyKey(public_key_bytes)
    try:
        verify_key.verify(message.encode("utf-8"), sig_bytes)
        return True
    except nacl.exceptions.BadSignatureError:
        return False


# ---------------------------------------------------------------------------
# Hedera — mirror-node-resolved pubkey, ED25519 or ECDSA secp256k1
# ---------------------------------------------------------------------------


# DER-encoded SubjectPublicKeyInfo prefixes Hedera mirror node returns.
# ED25519:  302a300506032b6570032100 || <32-byte raw pubkey>
# ECDSA secp256k1 (compressed): 302d300706052b8104000a032200 || <33-byte compressed>
_HEDERA_DER_ED25519_PREFIX = bytes.fromhex("302a300506032b6570032100")
_HEDERA_DER_ECDSA_PREFIX = bytes.fromhex("302d300706052b8104000a032200")

# Hedera Wallet Connect 2.x prefixes the message with this string before
# signing (similar to EIP-191 but for Hedera). We try both raw and prefixed
# framings to stay forward/backward compatible.
def _hedera_eip191_envelope(message: str) -> bytes:
    body = message.encode("utf-8")
    return b"\x19Hedera Signed Message:\n" + str(len(body)).encode() + body


def _parse_hedera_pubkey(hex_or_raw: str) -> tuple[str, bytes]:
    """Identify the Hedera key type from a hex-encoded pubkey.

    Accepts either:
      - DER-encoded SPKI (44 bytes ED25519 / 47 bytes ECDSA secp256k1) hex string
      - Raw 32-byte ED25519 pubkey hex
      - Raw 33-byte compressed ECDSA secp256k1 pubkey hex (0x02/0x03 prefix)

    Returns ('ed25519' | 'ecdsa_secp256k1', raw_pubkey_bytes).
    """
    s = hex_or_raw.strip().lower().removeprefix("0x")
    raw = bytes.fromhex(s)
    if raw.startswith(_HEDERA_DER_ED25519_PREFIX) and len(raw) == len(_HEDERA_DER_ED25519_PREFIX) + 32:
        return ("ed25519", raw[-32:])
    if raw.startswith(_HEDERA_DER_ECDSA_PREFIX) and len(raw) == len(_HEDERA_DER_ECDSA_PREFIX) + 33:
        return ("ecdsa_secp256k1", raw[-33:])
    if len(raw) == 32:
        return ("ed25519", raw)
    if len(raw) == 33 and raw[0] in (0x02, 0x03):
        return ("ecdsa_secp256k1", raw)
    raise ValueError(f"unrecognised Hedera pubkey format ({len(raw)} bytes)")


def _verify_hedera(
    account_id: str,
    message: str,
    signature: str,
    pubkey: str,
) -> bool:
    """Verify a Hedera signature.

    Hedera accounts can be ED25519 (native) or ECDSA secp256k1 (EVM-compat).
    The pubkey is supplied by the caller — typically fetched from the mirror
    node when the verify request arrives. Signatures are base64-encoded.

    For ECDSA secp256k1 we follow Hedera Wallet Connect's HIP-820 framing:
    keccak256("\\x19Hedera Signed Message:\\n<len>" + message) is the digest;
    signature is 64 bytes (r || s) or 65 (r || s || v). v is ignored — we
    test v=27 and v=28 internally if present.
    """
    import base64
    import hashlib
    import nacl.signing
    import nacl.exceptions

    try:
        sig_bytes = base64.b64decode(signature)
    except Exception:
        return False

    try:
        key_type, key_bytes = _parse_hedera_pubkey(pubkey)
    except ValueError:
        return False

    msg_bytes = message.encode("utf-8")
    envelope = _hedera_eip191_envelope(message)

    if key_type == "ed25519":
        verify_key = nacl.signing.VerifyKey(key_bytes)
        # Try every plausible framing — raw, "MX" prefix (some wallets reuse
        # the AlgoSigner convention), and the EIP-191-style Hedera envelope.
        for body in (msg_bytes, b"MX" + msg_bytes, envelope):
            try:
                verify_key.verify(body, sig_bytes)
                return True
            except nacl.exceptions.BadSignatureError:
                continue
        return False

    # ECDSA secp256k1 — use eth_keys (already a project dep via eth_account).
    try:
        from eth_keys import keys
        from eth_keys.exceptions import BadSignature
    except ImportError:
        logger.warning("wallet_auth.eth_keys_missing")
        return False

    try:
        # Decompress the 33-byte SEC1 pubkey to the 64-byte uncompressed form
        # eth_keys expects.
        from coincurve import PublicKey as _CCPub  # type: ignore
        uncompressed = _CCPub(key_bytes).format(compressed=False)[1:]
    except Exception:
        # coincurve isn't installed — fall back to a pure-Python decompress.
        uncompressed = _decompress_secp256k1(key_bytes)

    pub = keys.PublicKey(uncompressed)

    # Hedera's HIP-820 digest: keccak256(envelope).  Some signers omit the
    # envelope and digest the raw message; we accept both.
    digests = [
        _keccak256(envelope),
        _keccak256(msg_bytes),
        # Some library shipping Ethereum's exact prefix for cross-compat:
        _keccak256(b"\x19Ethereum Signed Message:\n" + str(len(msg_bytes)).encode() + msg_bytes),
    ]

    # eth_keys.Signature wants 65 bytes (r||s||v); if we only got 64 we try
    # both v values.
    if len(sig_bytes) == 64:
        sig_candidates = [sig_bytes + bytes([v]) for v in (0, 1, 27, 28)]
    elif len(sig_bytes) == 65:
        sig_candidates = [sig_bytes]
    else:
        return False

    for sig_raw in sig_candidates:
        # Normalise v to 0/1 for eth_keys (it accepts 0/1 and rejects 27/28).
        if sig_raw[64] >= 27:
            sig_raw = sig_raw[:64] + bytes([sig_raw[64] - 27])
        try:
            sig = keys.Signature(sig_raw)
        except (ValueError, BadSignature):
            continue
        for digest in digests:
            try:
                if sig.verify_msg_hash(digest, pub):
                    return True
            except (BadSignature, ValueError):
                continue
    return False


def _keccak256(data: bytes) -> bytes:
    # eth_account ships pysha3 → eth_utils provides keccak. Avoid the heavy
    # import by going through hashlib's sha3_256 isn't right; use eth_utils.
    from eth_utils.crypto import keccak
    return keccak(data)


def _decompress_secp256k1(compressed: bytes) -> bytes:
    """Decompress a 33-byte SEC1 secp256k1 pubkey to 64-byte (X || Y).

    Pure Python fallback used only when ``coincurve`` isn't available.
    Algorithm per SEC1 §2.3.4: y = sqrt((x^3 + 7) mod p) mod p, pick the
    parity matching the leading byte (0x02 = even Y, 0x03 = odd Y).
    """
    if len(compressed) != 33 or compressed[0] not in (0x02, 0x03):
        raise ValueError("not a compressed secp256k1 pubkey")
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    x = int.from_bytes(compressed[1:], "big")
    y_squared = (pow(x, 3, p) + 7) % p
    y = pow(y_squared, (p + 1) // 4, p)
    if (y & 1) != (compressed[0] & 1):
        y = p - y
    return x.to_bytes(32, "big") + y.to_bytes(32, "big")


async def fetch_hedera_account_key(
    account_id: str,
    *,
    network: str = "mainnet",
    timeout: float = 15.0,
) -> str | None:
    """Fetch the public key for a Hedera account from the mirror node REST API.

    Returns the DER-encoded SPKI as a hex string, or None if the account is
    missing or the lookup fails. Network is one of ``mainnet`` | ``testnet`` |
    ``previewnet``. Used by the recurr verify endpoint to bind a wallet to its
    on-chain key at challenge-redeem time.
    """
    import httpx

    base = {
        "mainnet": "https://mainnet-public.mirrornode.hedera.com",
        "testnet": "https://testnet.mirrornode.hedera.com",
        "previewnet": "https://previewnet.mirrornode.hedera.com",
    }.get(network, "https://mainnet-public.mirrornode.hedera.com")
    url = f"{base}/api/v1/accounts/{account_id}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, headers={"Accept": "application/json"})
        if r.status_code != 200:
            logger.warning(
                "wallet_auth.hedera_mirror_lookup_failed",
                extra={"account_id": account_id, "status": r.status_code},
            )
            return None
        data = r.json()
        key = data.get("key") or {}
        # Mirror node returns {"_type": "ED25519"|"ECDSA_SECP256K1", "key": "<hex>"}
        return key.get("key")
    except Exception as exc:  # noqa: BLE001 — return None for any error
        logger.warning(
            "wallet_auth.hedera_mirror_exception",
            extra={"account_id": account_id, "error": str(exc)},
        )
        return None


# ---------------------------------------------------------------------------
# Customer JWT
# ---------------------------------------------------------------------------


def issue_customer_jwt(
    *,
    wallet_address: str,
    chain: str,
    secret: str,
    ttl_seconds: int = DEFAULT_CUSTOMER_JWT_TTL_SECONDS,
    algorithm: str = "HS256",
    now_utc: datetime | None = None,
) -> tuple[str, datetime]:
    """Issue a short-lived JWT scoped to a customer's wallet — platform-wide.

    The wallet IS the identity. The token grants read across ALL tenants where
    that wallet appears in any `customers` row, plus the ability to cancel any
    of those tenants' subscriptions for that wallet. 30-min default TTL; to
    refresh, the customer signs another challenge.
    """
    now = now_utc or datetime.now(timezone.utc)
    exp = now + timedelta(seconds=ttl_seconds)
    # `sub` (subject) IS the wallet address. No tenant or customer_id in the
    # token — those are derived server-side at request time from the wallet.
    payload: dict[str, Any] = {
        "sub": wallet_address,
        "wal": wallet_address,
        "chn": chain,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "scope": "recurr.customer",
    }
    token = jwt.encode(payload, secret, algorithm=algorithm)
    return token, exp


@dataclass(frozen=True)
class CustomerSession:
    wallet_address: str
    chain: str
    issued_at: int
    expires_at: int


def decode_customer_jwt(
    token: str,
    *,
    secret: str,
    algorithms: list[str] | None = None,
) -> CustomerSession:
    """Verify + decode a customer JWT. Raises jwt.PyJWTError on any failure
    (expired, bad signature, wrong scope, malformed).

    Wallet-first: the JWT identifies a wallet only. Tenant + customer rows
    are resolved at request time from `customers.wallet_address`.
    """
    decoded = jwt.decode(
        token,
        secret,
        algorithms=algorithms or ["HS256"],
        options={"require": ["exp", "iat", "sub"]},
    )
    if decoded.get("scope") != "recurr.customer":
        raise jwt.InvalidTokenError("scope mismatch")
    return CustomerSession(
        wallet_address=str(decoded.get("wal") or decoded["sub"]),
        chain=str(decoded["chn"]),
        issued_at=int(decoded["iat"]),
        expires_at=int(decoded["exp"]),
    )
