{ fetchFromGitHub }:
let
  src = fetchFromGitHub {
    owner = "DavHau";
    repo = "mach-nix";
    rev = "f11fe2b5cb1cb2eae7074c7c52025ff09b1999a3";
    sha256 = "1jg9gdxw7xwpnhh2wg2qcvzhmhckzpkfbplw7q6vma55jwhgkf64";
  }
in
import src
