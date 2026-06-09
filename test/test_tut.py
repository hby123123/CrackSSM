import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--dataset", "TUT"] + [a for a in sys.argv[1:] if a != "--dataset"]
    from test.run_test import main
    main()
