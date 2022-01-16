mkdir -p HelloWorld.app/Contents/MacOS

clang++ -o HelloWorld.app/Contents/MacOS/HelloWorld HelloWorld.m -framework Cocoa
