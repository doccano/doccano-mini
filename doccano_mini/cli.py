import sys
from pathlib import Path

import streamlit.web.cli as stcli


def main():
    filepath = str(Path(__file__).parent.resolve() / "home.py")
    sys.argv = ["streamlit", "run", filepath, "--global.developmentMode=false"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
