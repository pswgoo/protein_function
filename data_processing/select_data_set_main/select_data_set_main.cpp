#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <set>
#include "data_class/protein_sequence.h"

int cnt=0;
using namespace std;

std::unordered_set<std::string> Spc = {"HALS3","HALVD","IGNH4","METJA","NITMS","PYRFU","SULSO","BACSU",
     "ECOLI","HELPY","MYCGE","PSEAE","PSEPK","PSESM","SALCH","SALTY","STRPN",
	 "ARATH","DANRE","DICDI","DROME","HUMAN","MOUSE","RAT","SCHPO","XENLA","YEAST"};
std::unordered_set<std::string> Exp = {"EXP","IDA","IPI","IMP","IGI","IEP","TAS","IC"};
std::unordered_set<std::string> Spc_cafa1 = {"BACSU","PSEAE","STRPN","ARATH","YEAST","XENLA","HUMAN","MOUSE","RAT","DICDI","ECOLI"};

void SaveBin(const string& file_in,const string& file_out) {
	ProteinSequenceSet seq;
	seq.ParseUniprotXml(file_in);
	seq.Save(file_out);
	
}
void SaveGoaBin(const string& file_in,const string& file_out) {
	ProteinSequenceSet seq;
	seq.ParseGoa(file_in);
	seq.Save(file_out);
}
void Savetop (const string& file_out,  ProteinSequenceSet &t) {
	ofstream out;
	out.open(file_out);
	int cnt=0,cnt_all=0;
	//for(int i = 0; i < t.protein_sequences().size() && i < 10 ; i++) {
	for(int i = 0; i < t.protein_sequences().size()  ; i++) {
	
		if(t.protein_sequences().at(i).go_terms().size() != 0) {
			cnt++;
			out<<t.protein_sequences().at(i).name()<<"_"<<t.protein_sequences().at(i).species()<<endl;
		}
		cnt_all++;
		//out<<t.protein_sequences().at(i).ToString();
// 		out<<t.protein_sequences().at(i).name()<<"_"<<t.protein_sequences().at(i).species()<<endl;
	}
	clog<<"GO !=  0 num = "<<cnt<<endl;
	clog<<"GO != 0  &&  GO=0 all  num = "<<cnt_all<<endl;
	out.close();
}
void UnionProtein(const string& file_uniport,const string& file_goa,ProteinSequenceSet& uniport) {
	//add same protein's goa  + add  new goa protein
	ProteinSequenceSet goa;
	uniport.Load(file_uniport);
	goa.Load(file_goa);
	std::set<string> goa_finded;
	std::set<string>::iterator leftit;
	unordered_map<string,int> umap;
	for(size_t i = 0; i < goa.protein_sequences().size() ; ++i) {
		string scut = goa.protein_sequences().at(i).name() + "_" + goa.protein_sequences().at(i).species();
               std::pair<std::string,int> pairtmp (scut,i);
		umap.insert(pairtmp);
	}
	unordered_map<string,int>::iterator it;
	for(size_t i = 0; i < uniport.protein_sequences().size(); ++i) {
		string sstay = uniport.protein_sequences().at(i).name() + "_" + uniport.protein_sequences().at(i).species();
            it = umap.find(sstay);
            if(it != umap.end())  {
			//GOA has this protein
			goa_finded.insert(sstay);
			unordered_map<int,int> ugo;
			unordered_map<int,int>::iterator uit;
			for(size_t j = 0; j<uniport.protein_sequences().at(i).go_terms().size(); ++j) {
				 std::pair<int,int> pair (uniport.protein_sequences().at(i).go_terms().at(j).id_,j);
				ugo.insert(pair);
			}
			for(size_t j = 0; j<goa.protein_sequences().at(it->second).go_terms().size(); ++j){
			
				int id = goa.protein_sequences().at(it->second).go_terms().at(j).id_;
				uit = ugo.find(id);
				if(uit == ugo.end()){
					
					uniport.protein_sequences().at(i).add_go_term(goa.protein_sequences().at(it->second).go_terms().at(j));
				}
			}
		}
	}
	for(size_t i = 0;i < goa.protein_sequences().size(); ++i){
	
		string left = goa.protein_sequences().at(i).name() + "_" + goa.protein_sequences().at(i).species();
		leftit = goa_finded.find(left);
		if(leftit == goa_finded.end()){
		
			uniport.add_protein_sequence(goa.protein_sequences().at(i));
		}
		
	}
	uniport.Save("union201101uniport_add_goa.proteinset");
	Savetop("union201101uniport_add_goa.txt",uniport);

}

void DeleteGo ( ProteinSequence& staygo, ProteinSequence& cutgo, ProteinSequence& leftgo) {

	unordered_set<int> setgo;
	unordered_set<int>::iterator it;
	for(size_t i = 0; i <cutgo.go_terms().size(); ++i) {
		setgo.insert(cutgo.go_terms().at(i).id_);
	}
	for(size_t i =0; i < staygo.go_terms().size(); ++i) {
	
		it = setgo.find(staygo.go_terms().at(i).id_);
		if(it == setgo.end()) {
			leftgo.set_name(staygo.name());
			leftgo.set_species(staygo.species());
			leftgo.set_accessions(staygo.accessions());
			leftgo.set_ref_pmids(staygo.ref_pmids());
			leftgo.set_sequence(staygo.sequence());
			leftgo.add_go_term(staygo.go_terms().at(i));
		}
	}
}

/*
 * staygo-cutgo => left
 */
void AllProtein(  ProteinSequenceSet stayp ,   ProteinSequenceSet cutp,ProteinSequenceSet& leftp) {

	unordered_map<string,int> setp;
	for(size_t i = 0; i < cutp.protein_sequences().size() ; ++i) {
		string scut = cutp.protein_sequences().at(i).name() + "_" + cutp.protein_sequences().at(i).species();
        std::pair<std::string,int> pairtmp (scut,i);
		setp.insert(pairtmp);
	}
	unordered_map<string,int>::iterator it;
	for(size_t i = 0; i < stayp.protein_sequences().size(); ++i) {
		string sstay = stayp.protein_sequences().at(i).name() + "_" + stayp.protein_sequences().at(i).species();
            it = setp.find(sstay);
            if(it != setp.end())  {
            //find it
			ProteinSequence tmp;
			DeleteGo(stayp.protein_sequences().at(i) ,cutp.protein_sequences().at(it->second),tmp);
			leftp.add_protein_sequence(tmp);
		} 
		else {
			leftp.add_protein_sequence(stayp.protein_sequences().at(i));
        }
	}
}
void NewProtein( ProteinSequenceSet& stayp ,   ProteinSequenceSet& cutp,ProteinSequenceSet& leftp) {

	int BACSU=0,PSEAE=0,STRPN=0,ARATH=0,YEAST=0,XENLA=0,HUMAN=0,MOUSE=0,RAT =0,DICDI=0,ECOLI=0;
	int new_c=0,new_p=0,new_f=0;
	unordered_map<string,int> setp;
	for(size_t i = 0; i < cutp.protein_sequences().size() ; ++i) {
		string scut = cutp.protein_sequences().at(i).name() + "_" + cutp.protein_sequences().at(i).species();
               std::pair<std::string,int> pairtmp (scut,i);
		setp.insert(pairtmp);
	}
	unordered_map<string,int>::iterator it;
	for(size_t i = 0; i < stayp.protein_sequences().size(); ++i) {
		string sstay = stayp.protein_sequences().at(i).name() + "_" + stayp.protein_sequences().at(i).species();
                it = setp.find(sstay);
                if(it != setp.end())  {
               //find protein,but GO == 0
			
			if(cutp.protein_sequences().at(it->second).go_terms().size() == 0 && stayp.protein_sequences().at(i).go_terms().size() != 0){
				leftp.add_protein_sequence(stayp.protein_sequences().at(i));
				if(stayp.protein_sequences().at(i).species() =="BACSU") BACSU++;
				if(stayp.protein_sequences().at(i).species() =="PSEAE") PSEAE++;
				if(stayp.protein_sequences().at(i).species() =="STRPN") STRPN++;
				if(stayp.protein_sequences().at(i).species() =="ARATH") ARATH++;
				if(stayp.protein_sequences().at(i).species() =="YEAST") YEAST++;
				if(stayp.protein_sequences().at(i).species() =="XENLA") XENLA++;
				if(stayp.protein_sequences().at(i).species() =="HUMAN") HUMAN++;
				if(stayp.protein_sequences().at(i).species() =="MOUSE") MOUSE++;
				if(stayp.protein_sequences().at(i).species() =="RAT") RAT++;
				if(stayp.protein_sequences().at(i).species() =="DICDI") DICDI++;
				if(stayp.protein_sequences().at(i).species() =="ECOLI") ECOLI++;
			}
			
		} 
		else {
		
			if(stayp.protein_sequences().at(i).go_terms().size() != 0){
				leftp.add_protein_sequence(stayp.protein_sequences().at(i));
				if(stayp.protein_sequences().at(i).species() =="BACSU") BACSU++;
				if(stayp.protein_sequences().at(i).species() =="PSEAE") PSEAE++;
				if(stayp.protein_sequences().at(i).species() =="STRPN") STRPN++;
				if(stayp.protein_sequences().at(i).species() =="ARATH") ARATH++;
				if(stayp.protein_sequences().at(i).species() =="YEAST") YEAST++;
				if(stayp.protein_sequences().at(i).species() =="XENLA") XENLA++;
				if(stayp.protein_sequences().at(i).species() =="HUMAN") HUMAN++;
				if(stayp.protein_sequences().at(i).species() =="MOUSE") MOUSE++;
				if(stayp.protein_sequences().at(i).species() =="RAT") RAT++;
				if(stayp.protein_sequences().at(i).species() =="DICDI") DICDI++;
				if(stayp.protein_sequences().at(i).species() =="ECOLI") ECOLI++;
			}
                }
	}
			clog<<"BACSU = "<<BACSU<<endl;
			clog<<"PSEAE ="<<PSEAE<<endl;
			clog << "STRPN ="<<STRPN<<endl;
			clog << "ARATH ="<<ARATH<<endl;
			clog << "YEAST ="<<YEAST<<endl;
			clog << "XENLA ="<<XENLA<<endl;
			clog << "HUMAN ="<<HUMAN<<endl;
			clog << "MOUSE ="<<MOUSE<<endl;
			clog << "RAT ="<<RAT<<endl;
			clog << "DICDI ="<<DICDI<<endl;
			clog << "ECOLI ="<<ECOLI<<endl;
	
}
void NewProteinMF(   ProteinSequenceSet& stayp ,    ProteinSequenceSet& cutp,ProteinSequenceSet& leftp) {

	int new_c=0,new_p=0,new_f=0;
	unordered_map<string,int> setp;
	for(size_t i = 0; i < cutp.protein_sequences().size() ; ++i) {
		string scut = cutp.protein_sequences().at(i).name() + "_" + cutp.protein_sequences().at(i).species();
               std::pair<std::string,int> pairtmp (scut,i);
		setp.insert(pairtmp);
	}
	unordered_map<string,int>::iterator it;
	for(size_t i = 0; i < stayp.protein_sequences().size(); ++i) {
		int cc=0,mf=0,bp=0;
		for(int k=0;k<stayp.protein_sequences().at(i).go_terms().size();++k) {
			if(stayp.protein_sequences().at(i).go_terms().at(k).term_.at(0) == 'C') cc++;
			if(stayp.protein_sequences().at(i).go_terms().at(k).term_.at(0) == 'P') bp++;
			if(stayp.protein_sequences().at(i).go_terms().at(k).term_.at(0) == 'F') mf++;
		}
		string sstay = stayp.protein_sequences().at(i).name() + "_" + stayp.protein_sequences().at(i).species();
                it = setp.find(sstay);
                if(it != setp.end())  {
               //find protein,but new mf != 0, old = 0;
			if(cutp.protein_sequences().at(it->second).go_terms().size() == 0 && mf != 0){
				leftp.add_protein_sequence(stayp.protein_sequences().at(i));
			}
		} 
		else {
			if(mf != 0){
				leftp.add_protein_sequence(stayp.protein_sequences().at(i));
			}
		}
	}

	//clog<<"cnt="<<cnt<<" !"<<endl;
}
void NewProteinBP(   ProteinSequenceSet& stayp ,    ProteinSequenceSet& cutp,ProteinSequenceSet& leftp) {

	int new_c=0,new_p=0,new_f=0;
	unordered_map<string,int> setp;
	for(size_t i = 0; i < cutp.protein_sequences().size() ; ++i) {
		string scut = cutp.protein_sequences().at(i).name() + "_" + cutp.protein_sequences().at(i).species();
               std::pair<std::string,int> pairtmp (scut,i);
		setp.insert(pairtmp);
	}
	unordered_map<string,int>::iterator it;
	for(size_t i = 0; i < stayp.protein_sequences().size(); ++i) {
		int cc=0,mf=0,bp=0;
		for(int k=0;k<stayp.protein_sequences().at(i).go_terms().size();++k) {
			if(stayp.protein_sequences().at(i).go_terms().at(k).term_.at(0) == 'C') cc++;
			if(stayp.protein_sequences().at(i).go_terms().at(k).term_.at(0) == 'P') bp++;
			if(stayp.protein_sequences().at(i).go_terms().at(k).term_.at(0) == 'F') mf++;
		}
		string sstay = stayp.protein_sequences().at(i).name() + "_" + stayp.protein_sequences().at(i).species();
                it = setp.find(sstay);
                if(it != setp.end())  {
               //find protein,but new bp != 0, old = 0;
			if(cutp.protein_sequences().at(it->second).go_terms().size() == 0 && bp != 0){
				leftp.add_protein_sequence(stayp.protein_sequences().at(i));
			}
		} 
		else {
		
			if(bp != 0){
				leftp.add_protein_sequence(stayp.protein_sequences().at(i));
			}
		}
	}

	//clog<<"cnt="<<cnt<<" !"<<endl;
}
void NewGo(  ProteinSequenceSet stayp ,   ProteinSequenceSet cutp, ProteinSequenceSet& leftp) {

	int BACSU=0,PSEAE=0,STRPN=0,ARATH=0,YEAST=0,XENLA=0,HUMAN=0,MOUSE=0,RAT =0,DICDI=0,ECOLI=0;
	unordered_map<string,int> setp;
	for(size_t i = 0; i < cutp.protein_sequences().size() ; ++i) {
		string scut = cutp.protein_sequences().at(i).name() + "_" + cutp.protein_sequences().at(i).species();
               std::pair<std::string,int> pairtmp (scut,i);
		setp.insert(pairtmp);
	}
	unordered_map<string,int>::iterator it;
	for(size_t i = 0; i < stayp.protein_sequences().size(); ++i) {
		string sstay = stayp.protein_sequences().at(i).name() + "_" + stayp.protein_sequences().at(i).species();
                it = setp.find(sstay);
		int staybp=0,staymf=0,staycc=0;
		for(int j = 0; j < stayp.protein_sequences().at(i).go_terms().size(); ++j) {
				char s =  stayp.protein_sequences().at(i).go_terms().at(j).term_.at(0);
				if( s == 'P')  staybp++;
				else if (s == 'F') staymf++;
				else if(s == 'C') staycc++;
		}
		string p1="AFAM",p2="HUMAN";
			if( stayp.protein_sequences().at(i).name() == p1 &&  stayp.protein_sequences().at(i).species() == p2)
				clog<<"staymf="<<staymf<<endl;
			
                if(it != setp.end())  {
               //find protein,but GO == 0, must update
			
			int cutbp=0,cutmf=0,cutcc=0;
			for(int j = 0; j < cutp.protein_sequences().at(it->second).go_terms().size(); ++j) {
			
				char s =  cutp.protein_sequences().at(it->second).go_terms().at(j).term_.at(0);
				if( s == 'P')  cutbp++;
				else if (s == 'F') cutmf++;
				else if (s == 'C') cutcc++;
			}
			//if(((cutbp == 0 &&staybp !=0) ||(cutcc == 0 && cutcc != 0) || (cutmf == 0 && cutmf != 0)) && !(cutbp==0 &&cutcc==0&&cutmf ==0)){
			if(cutmf == 0 && cutbp== 0  &&cutcc == 0 &&staymf !=0) {
				if( stayp.protein_sequences().at(i).name() == p1 &&  stayp.protein_sequences().at(i).species() == p2)
				clog<<"test!!!!!!!!!"<<endl;
				leftp.add_protein_sequence(stayp.protein_sequences().at(i));
			}
		} 
		else {
			if(staymf != 0){
				leftp.add_protein_sequence(stayp.protein_sequences().at(i));
			}
		}
	}
	
}
/**
 * @brief save bin 
 */
void SelectTestSet(const string&file_new,const string& file_old,const string& file_goa,const string& file_out,int flag) {
	ProteinSequenceSet union1401,uniport1409,left,leftnewgo;
	uniport1409.Load(file_new);
	uniport1409.FilterGoByEvidence(Exp);
// 	uniport1409.FilterProteinSequenceBySpicies(Spc);
// 	uniport1409.FilterGoByEvidence(Exp_cafa1);
	uniport1409.FilterProteinSequenceBySpicies(Spc_cafa1);
	clog<<"201409 filter left protein number =  "<<uniport1409.protein_sequences().size()<<" !"<<endl;
	uniport1409.Save("uniport201201_filter.proteinset");
	Savetop("uniport201201_filter.txt",uniport1409);

	UnionProtein(file_old,file_goa,union1401);
	clog<<"union left protein number =  "<<union1401.protein_sequences().size()<<" !"<<endl;
	union1401.FilterGoByEvidence(Exp);
// 	union1401.FilterProteinSequenceBySpicies(Spc);
// 	union1401.FilterGoByEvidence(Exp_cafa1);
	union1401.FilterProteinSequenceBySpicies(Spc_cafa1);
	clog<<"union filter left protein number =  "<<union1401.protein_sequences().size()<<" !"<<endl;
	union1401.Save("union201401_filter.proteinset");
	Savetop("union201101_filter.txt",union1401);
	if(flag == 1){
		//new protein 
		NewProtein(uniport1409,union1401,left);
		left.Save(file_out);
		//left.Save("CAFA2_answer_new_protein.proteinset");
		//Savetop("CAFA2_answer_new_protein.txt",left);
		//Savetop("test.txt",left);
		clog<<" new protein number =  "<<left.protein_sequences().size()<<" !"<<endl;
	}
	else if(flag == 2) {
		//New Go
		NewGo(uniport1409,union1401,leftnewgo);
		leftnewgo.Save(file_out);
		//Savetop("CAFA2_answer_new_go.txt",leftnewgo);
		clog<<" new GO number =  "<<leftnewgo.protein_sequences().size()<<" !"<<endl;
	}
}

void Cafa1TestSet(const string&file_new,const string& file_old,const string& file_goa){
	ProteinSequenceSet union1101,uniport1112,left,leftnewgo;
	uniport1112.Load(file_new);
	uniport1112.FilterGoByEvidence(Exp);
	uniport1112.FilterProteinSequenceBySpicies(Spc_cafa1);
	clog<<"201112 filter Exp_cafa1 and Spc_cafa1    left protein number =  "<<uniport1112.protein_sequences().size()<<" !"<<endl;
	uniport1112.Save("uniport201112_filter.proteinset");
	Savetop("uniport201112_filter.txt",uniport1112);

	UnionProtein(file_old,file_goa,union1101);
	clog<<"union left protein number =  "<<union1101.protein_sequences().size()<<" !"<<endl;
	union1101.FilterGoByEvidence(Exp);
	union1101.FilterProteinSequenceBySpicies(Spc_cafa1);
	clog<<"union filter left protein number =  "<<union1101.protein_sequences().size()<<" !"<<endl;
	union1101.Save("union201101_filter.proteinset");
	Savetop("union201112_filter.txt",union1101);
		//new protein 
		NewProtein(uniport1112,union1101,left);
		left.Save("201112cut201101.proteinset");
		clog<<" new protein number =  "<<left.protein_sequences().size()<<" !"<<endl;
		//new mf 
// 		ProteinSequenceSet leftmf,leftbp;
// 		NewProteinMF(uniport1112,union1101,leftmf);
// 		leftmf.Save("201112cut201101mf.proteinset");
// 		clog<<" new mf number =  "<<leftmf.protein_sequences().size()<<" !"<<endl;
// 		//new bp 
// 		NewProteinBP(uniport1112,union1101,leftbp);
// 		leftbp.Save("201112cut201101bp.proteinset");
// 		clog<<" new bp number =  "<<leftbp.protein_sequences().size()<<" !"<<endl;
}
void testload()
{
	ProteinSequenceSet u201112,u;
	int t = u201112.ParseDat("/protein/raw_data/uniprot_sprot_201409.dat");
	u.ParseUniprotXml("/protein/raw_data/uniprot_sprot_201409.xml");
	
	int  cnt=0,nofind=0;
	unordered_map<string,int> map_u;
	for(int i=0;i<u.protein_sequences().size();++i){
		string s=u.protein_sequences().at(i).name()+"_"+u.protein_sequences().at(i).species();
		std::pair<std::string,int> pairtmp (s,i);
		map_u.insert(pairtmp);
	}
	unordered_map<string,int>::iterator it;
	for(size_t i = 0; i < u201112.protein_sequences().size(); ++i) {
		string sstay = u201112.protein_sequences().at(i).name() + "_" + u201112.protein_sequences().at(i).species();
                it = map_u.find(sstay);
                if(it != map_u.end())  {

			if(u.protein_sequences().at(it->second).go_terms().size() !=u201112.protein_sequences().at(i).go_terms().size()) cnt++;

			for(int k=0;k<u.protein_sequences().at(it->second).go_terms().size();++k){
				
				if(u.protein_sequences().at(it->second).go_terms().at(k).evidence_ !=u201112.protein_sequences().at(i).go_terms().at(k).evidence_) 
				{
					cnt++;
					clog<<u.protein_sequences().at(it->second).name()<<"_"<<u.protein_sequences().at(it->second).species()<<endl;
					clog<<u.protein_sequences().at(it->second).go_terms().at(k).evidence_ <<endl;
					clog<<u201112.protein_sequences().at(i).go_terms().at(k).evidence_<<endl;
				}
			}
		}
		else nofind++;
	}
	clog<<"cnt=\t"<<cnt<<endl;
	clog<<"notfind=\t"<<nofind<<endl;
}
int main(int argc, char * argv[]) {
	ProteinSequenceSet a,b,c,d,goa1401;
 	
// 	Savetop("goa_201101.txt",c);
 	//SaveBin("/protein/raw_data/uniprot_sprot_201201.xml","uniprot_sprot_201201.proteinset");
// 	SaveGoaBin("/protein/raw_data/gene_association.goa_uniprot_noiea_protein_only_2011","goa_201101.proteinset");
 	//SelectTestSet("uniprot_sprot_201201.uniprot_sprot_201201","uniprot_sprot_201101.proteinset","goa_201101.proteinset","201201cut201101mf.proteinset",2);
	//SelectTestSet("uniprot_sprot_201504.proteinset","uniprot_sprot_201401.proteinset","goa_201401.proteinset","201504_filter_201401_new_protein.proteinset",1);
	//Cafa1TestSet("uniprot_sprot_201201.proteinset","uniprot_sprot_201101.proteinset","goa_201101.proteinset");
	//a.Load("201112cut201101.proteinset");
	//Savetop("201112cut201101.txt",a);
	
	
	ProteinSequenceSet u201112;
	//int t = u201112.ParseDat("/protein/raw_data/uniprot_sprot_201112.dat");
 	
	u201112.Load("uniprot_sprot_201112.proteinset");
	
// 	u201112.SaveToString("uniprot_sprot_201112.txt");
	u201112.FilterProteinSequenceBySpicies(Spc_cafa1);
	u201112.FilterGoCc();
	u201112.FilterGoByEvidence(Exp);
 	clog<<"201112 filter left protein number =  "<<u201112.protein_sequences().size()<<" !"<<endl;
 	Savetop("uniport201112_filter_id.txt",u201112);
	u201112.SaveToString("uniport201112_test2.txt");


// 	b.Load("201112cut201101mf.proteinset");
// 	Savetop("201201cut201101mf.txt",b);
// 	
// 	c.Load("201112cut201101bp.proteinset");
// 	Savetop("201112cut201101bp.txt",c);
	//b.Load("201201cut201101mf.proteinset");
	//Savetop("201201cut201101mf.txt",b);
	/*********************************************************************/
// 	a.ParseUniprotXml(argv[1]);
// 	if(strcmp(argv[3] , "1") == 0)
// 	{
// 		a.FilterGoByEvidence(Exp);
// 		clog<<"only leave 8 evidences, total "<<a.protein_sequences().size()<<" protein sequences !"<<endl;
// 	}
// 	if(strcmp(argv[4] , "1") == 0)
// 	{
// 		a.FilterProteinSequenceBySpicies(Spc);
// 		clog<<"only leave 27 spiceis, total "<<a.protein_sequences().size()<<" protein sequences !"<<endl;
// 	}
// 	if(strcmp(argv[5] , "1") == 0)
// 	{
// 		ProteinSequenceSet cp;
// 		for(int i=0; i<a.protein_sequences().size(); ++i) {
// 		
// 			if(a.protein_sequences().at(i).go_terms().size() != 0)
// 				cp.add_protein_sequence(a.protein_sequences().at(i));
// 		}
// 		a=cp;
// 		clog<<"only leave GO != 0 proteins, total "<<a.protein_sequences().size()<<" protein sequences !"<<endl;
// 	}
// 	if(strcmp(argv[6],"0") == 0){
// 		a.SaveToFasta(argv[2]);
// 		clog<<"Finish save data to Fasta! "<<endl;
// 	}
// 	if(strcmp(argv[6],"1") == 0){
// 		ofstream out;
// 		out.open(argv[2]);
// 		for(int i = 0; i < a.protein_sequences().size()  ; i++) {
// 			out<<a.protein_sequences().at(i).ToString();
// 		}
// 		out.close();
// 		clog<<"Finish save data to string! "<<endl;
// 	}
// 	
// 	if(strcmp(argv[6],"2") == 0){
// 		
// 		a.Save(argv[2]);
// 		clog<<"Finish save data to BIN! "<<endl;
// 	}
	/***************************************************************/
	return 0;
}
 
