all: clean  compile clear run


compile:

	g++ -o camera Image_process.cpp Filter.cpp `pkg-config --cflags --libs opencv4`


run:

	./camera

clean:

	rm -f camera

clear:

	clear