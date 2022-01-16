mkdir -p OrthoTriangle.app/Contents/MacOS

clang++ -o OrthoTriangle.app/Contents/MacOS/OrthoTriangle OrthographicTriangle.mm -framework Cocoa -framework QuartzCore -framework OpenGL
