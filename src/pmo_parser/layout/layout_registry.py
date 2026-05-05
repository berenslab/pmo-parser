"""
Registry holding the available :class:`PDFLayout` implementations.

Layout classes register themselves here on import via the ``@register``
decorator so callers can resolve a backend by name without hard-coded
imports.
"""

from pmo_parser.registry import Registry as _Registry

LAYOUT_REGISTRY = _Registry()
