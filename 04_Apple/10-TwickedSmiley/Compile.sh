mkdir -p TwickedSmiley.app/Contents/MacOS

clang++ -o TwickedSmiley.app/Contents/MacOS/TwickedSmiley TwickedSmiley.mm -framework Cocoa -framework QuartzCore -framework OpenGL
