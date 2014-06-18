CC = g++
CFLAGS = -g -Wall -I. -I./
LDFLAGS = 
BIN = NB
OBJS = main.o 

all: $(BIN)

$(BIN):	$(OBJS)
	$(CC) $^ -o $@
	
clean:
	$(RM) *.o  $(BIN)
