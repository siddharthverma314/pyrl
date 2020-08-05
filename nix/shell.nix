let
  pkgs = import ./nixpkgs.nix;
in
with pkgs;
mkShell {
  buildInputs = [
    (python38.withPackages (ps: with ps; [
      mujoco-py_gpu
      gym
    ]))
  ];
}
