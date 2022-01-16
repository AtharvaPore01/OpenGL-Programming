mkdir -p Two3DShapesWithTexture.app/Contents/MacOS

clang++ -o Two3DShapesWithTexture.app/Contents/MacOS/Two3DShapesWithTexture Two3DShapesWithTexture.mm -framework Cocoa -framework QuartzCore -framework OpenGL
