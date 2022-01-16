mkdir -p GeometryShader.app/Contents/MacOS

clang++ -o GeometryShader.app/Contents/MacOS/GeometryShader GeometryShader.mm -framework Cocoa -framework QuartzCore -framework OpenGL
