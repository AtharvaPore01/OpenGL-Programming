mkdir -p AllGeometryGraphPaper.app/Contents/MacOS

clang++ -o AllGeometryGraphPaper.app/Contents/MacOS/AllGeometryGraphPaper AllGeometryGraphPaper.mm -framework Cocoa -framework QuartzCore -framework OpenGL
