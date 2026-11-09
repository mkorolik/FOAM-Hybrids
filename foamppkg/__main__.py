"""
Foam module interface.

Usage:
    foam version 
    foam test

"""

if __name__ == "__main__":

    import sys
    import foamppkg

    if sys.argv[-1] == 'version':
        sys.exit(foamppkg.__version__)
    
    if sys.argv[-1] == 'test':
        import pytest, os
        from pathlib import Path
        module_dir = str(Path(os.path.abspath(foamppkg.__file__)).parent.parent)
        pytest.main([module_dir])

