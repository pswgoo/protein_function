#include <iostream>
#include <string>
#include <map>
#include <fstream>

#include "data_class/protein_sequence.h"

using namespace std;

std::unordered_set<std::string> Exp = {"EXP","IDA","IPI","IMP","IGI","IEP","TAS","IC"};
std::unordered_set<std::string> Spc_cafa1 = {"BACSU","PSEAE","STRPN","ARATH","YEAST","XENLA","HUMAN","MOUSE","RAT","DICDI","ECOLI"};

int main() {

	ifstream ys("201112.txt");
	ifstream qy("uniport201112_filter_id.txt");
	ofstream out("out.txt");
        map<string,int> mq;
	map<string,int>::iterator pos;
	string s;
	int k = 0;
	int cnt = 0;
        while(getline(qy,s)){
                 std::pair<std::string,int> pairtmp (s,k);
		 mq.insert(pairtmp);
        }
        while(getline(ys ,s)){
	
		pos = mq.find(s);
		if(pos == mq.end())
		{
			 cnt++;
			out<<s<<endl;
		}
// 		else cnt++;
	}
	clog<<cnt<<endl;
	qy.close();
	ys.close();
	out.close();
}
