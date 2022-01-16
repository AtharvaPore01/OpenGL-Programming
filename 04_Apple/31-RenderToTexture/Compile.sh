mkdir -p RenderToTexture.app/Contents/MacOS

clang++ -o RenderToTexture.app/Contents/MacOS/RenderToTexture RenderToTexture.mm -framework Cocoa -framework QuartzCore -framework OpenGL
