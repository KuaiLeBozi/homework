CXX = g++
CXXFLAGS = -O3 -mavx512f -fopenmp 
SRCS = winograd.cc driver.cc
OBJS = $(SRCS:.cc=.o)
TARGET = winograd

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(OBJS) $(TARGET)