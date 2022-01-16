mkdir -p Two2DShapesAnimating.app/Contents/MacOS

clang++ -o Two2DShapesAnimating.app/Contents/MacOS/Two2DShapesAnimating Two2DShapesAnimating.mm -framework Cocoa -framework QuartzCore -framework OpenGL
