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
        packages = [ pkgs.mold pkgs.ffmpeg-full ];

        inherit LD_LIBRARY_PATH;
      };
    };
}

# ffmpeg -i $INPUT_FILENAME \
# -vf "fps=$OUTPUT_FPS,scale=$OUTPUT_WIDTH:-1:flags=lanczos" \
# -vcodec libwebp -lossless 0 -compression_level 6 \
# -q:v $OUTPUT_QUALITY -loop $NUMER_OF_LOOPS \
# -preset picture -an -vsync 0 $OUTPUT_FILENAME

# # Change these placeholders:
# # * $INPUT_FILENAME - path to the input video.
# # * $OUTPUT_FPS - ouput frames per second. Start with `10`.
# # * $OUTPUT_WIDTH - output width in pixels. Aspect ratio is maintained.
# # * $OUTPUT_QUALITY - quality of the WebP output. Start with `50`.
# # * $NUMBER_OF_LOOPS - use `0` to loop forever, or a specific number of loops.
# # * $OUTPUT_FILENAME - the name of the output animated WebP.
