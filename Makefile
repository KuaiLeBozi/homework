CXX = g++
CXXFLAGS = -O3 -fopenmp -march=native 
SRCS = winograd.cc driver.cc
OBJS = $(SRCS:.cc=.o)
TARGET = winograd

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(OBJS) $(TARGET)
