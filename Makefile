PROG=rmacoll

PROG_OBJ=$(PROG).o broadcast_linear.o broadcast_binomial.o rmautils.o 

INCLUDE_PATH=./

CXX=mpicxx
CXXFLAGS=-Wall -I$(INCLUDE_PATH) -pthread \
		 -lboost_system -lboost_date_time -lboost_thread \
		 -L/usr/lib64

all: $(PROG)

$(PROG): $(PROG_OBJ) 
	$(CXX) $(CXXFLAGS) $^ -o $@ 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

clean:
	rm -rf $(PROG) $(PROG_OBJ) 
	rm -rf stdout
	rm -rf stdout
	rm -rf mpitask.o*
