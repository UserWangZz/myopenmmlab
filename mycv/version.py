__version__ = '1.0.0'

version_info = tuple(int(x) for x in __version__.split('.')[:3])

__all__ = ['__version__', 'version_info']