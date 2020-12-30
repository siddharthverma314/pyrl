{ pkgs, pkg, pythonpaths ? [""] }:
pkgs.mkShell {
  buildInputs = with pkgs; [
    nodePackages.pyright
    ripgrep
    (python3.withPackages (ps: with ps; [
      black
      pkg
      ipdb
      rope
      pyflakes
      pytest
      tensorflow-tensorboard_2
    ]))
  ];
  shellHook = ''
    export PYTHONPATH=${pkgs.lib.concatStringsSep ":" (map (s: "$PWD/${s}") pythonpaths)}
  '';
}
