{
  outputs = { nixpkgs, self }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib";
    in
    {
      devShell."${system}" = pkgs.mkShell {
        inputsFrom = [ ];
        packages = [ pkgs.mold pkgs.ffmpeg ];

        inherit LD_LIBRARY_PATH;
      };
    };
}
