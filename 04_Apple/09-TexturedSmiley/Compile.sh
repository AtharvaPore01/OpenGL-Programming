mkdir -p TexturedSmiley.app/Contents/MacOS

clang++ -o TexturedSmiley.app/Contents/MacOS/TexturedSmiley TexturedSmiley.mm -framework Cocoa -framework QuartzCore -framework OpenGL
