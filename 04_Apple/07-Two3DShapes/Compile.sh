mkdir -p Two3DShapes.app/Contents/MacOS

clang++ -o Two3DShapes.app/Contents/MacOS/Two3DShapes Two3DShapes.mm -framework Cocoa -framework QuartzCore -framework OpenGL
