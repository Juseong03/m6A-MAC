"""Frozen RiNALMo with a MAC U-Net head."""
from .m6a_mac import M6AMAC, FrozenMACHead

__all__ = ["M6AMAC", "FrozenMACHead"]
