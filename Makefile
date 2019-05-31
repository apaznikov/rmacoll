PROG=rmacoll

PROG_OBJ=$(PROG).o rmautils.o broadcast_linear.o broadcast_binomial.o

INCLUDE_PATH=./

CXX=mpicxx
CXXFLAGS=-Wall -I$(INCLUDE_PATH) -pthread \
		 -lboost_system -lboost_date_time -lboost_thread

all: $(PROG)

$(PROG): $(PROG_OBJ) 
	$(CXX) $(CXXFLAGS) $^ -o $@ 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

clean:
	rm -rf $(PROG) $(PROG_OBJ) 
	rm -rf stdout
