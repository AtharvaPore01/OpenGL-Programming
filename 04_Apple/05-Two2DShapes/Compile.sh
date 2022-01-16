mkdir -p Two2DShapes.app/Contents/MacOS

clang++ -o Two2DShapes.app/Contents/MacOS/Two2DShapes Two2DShapes.mm -framework Cocoa -framework QuartzCore -framework OpenGL
