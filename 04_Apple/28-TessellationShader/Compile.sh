mkdir -p TessellationShader.app/Contents/MacOS

clang++ -o TessellationShader.app/Contents/MacOS/TessellationShader TessellationShader.mm -framework Cocoa -framework QuartzCore -framework OpenGL
