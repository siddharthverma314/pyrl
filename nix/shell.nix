let
  pkgs = import ./nixpkgs.nix;
in
with pkgs;
mkShell {
  buildInputs = [
    (python38.withPackages (ps: with ps; [
      scikit-video
    ]))
  ];
}
