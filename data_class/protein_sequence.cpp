#include "protein_sequence.h"

#include <iostream>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>
#include <sstream>

#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

using namespace std;
using namespace boost;

AminoType GetAminoType(const std::string &x) {
	for	(int name = AminoType::NON; name <= AminoType::VAL; ++name)
		if (kTypeString[name] == x || (x.length() == 1 && x[0] == kTypeChar[name]))
			return static_cast<AminoType>(name);
	return AminoType::NON;
}

static const boost::property_tree::ptree& EmptyPtree() {
	static boost::property_tree::ptree t;
	return t;
}

std::string ProteinSequence::ToString() const {
	string ret_value("Protein sequence: \n");

	ret_value += "name: " + name() + "\n";
	ret_value += "species: " + species() + "\n";
	ret_value += "accessions:";
	for (auto &u : accessions())
		ret_value += " " + u;
	ret_value += "\n";
	ret_value += "GO: " + to_string(go_terms().size()) + "\n";
	for (auto &u : go_terms()) {
		ret_value += "GO_id: " + to_string(u.id_) + " ";
		ret_value += "GO_term: " + u.term_ + " ";
		ret_value += "GO_evidence: " + u.evidence_ + "\n";
	}
	ret_value += "ref pmids: " + to_string(ref_pmids().size()) + "\n";
	for (auto u : ref_pmids())
		ret_value += " " + to_string(u);
	ret_value += "\n";
	ret_value += "sequence: " + to_string(sequence().size()) + " ";
	for (auto u : sequence())
		ret_value += Get1LetterAminoName(u);
	ret_value += "\n";
	return ret_value;
}

std::size_t ProteinSequence::ParseUniprotPtree(const boost::property_tree::ptree & rt_tree) {
	using namespace boost::property_tree;

	if (rt_tree.empty())
		return 0;

	BOOST_FOREACH(const ptree::value_type& v, rt_tree) {
		if (v.first == "accession") {
			add_accession(v.second.data());
		}
		else if (v.first == "name") {
			vector<string> tokens;
			boost::split(tokens, v.second.data(), is_any_of("_"));
			if (tokens.size() != 2)
				cerr << "Warning: name tokens not equal to 2" << endl;
			else {
				set_name(tokens[0]);
				set_species(tokens[1]);
			}
		}
		else if (v.first == "reference") {
			const ptree &ptree_citation = v.second.get_child("citation", EmptyPtree());
			BOOST_FOREACH(const ptree::value_type& v2, ptree_citation) {
				if (v2.first == "dbReference") {
					string db = boost::to_upper_copy(v2.second.get("<xmlattr>.type", string()));
					if (db == "PUBMED") {
						int pmid = v2.second.get("<xmlattr>.id", -1);
						if (pmid < 0)
							cerr << "Warning: pmid not valid" << endl;
						else
							add_ref_pmid(pmid);
					}
				}
			}
		}
		else if (v.first == "dbReference") {
			string db = boost::to_upper_copy(v.second.get("<xmlattr>.type", string()));
			if (db == "GO") {
				string str_id = v.second.get("<xmlattr>.id", string());
				size_t pos = str_id.rfind(":") + 1;
				str_id = str_id.substr(pos);
				if (str_id.empty()) {
					cerr << "Warning: GO id is empty" << endl;
				}
				else {
					ProteinSequence::GOType go;
					BOOST_FOREACH(const ptree::value_type& v2, v.second) {
						string type = v2.second.get("<xmlattr>.type", string());
						if (type == "term")
							go.term_ = v2.second.get("<xmlattr>.value", string());
						else if (type == "evidence")
							go.evidence_ = v2.second.get("<xmlattr>.value", string());
						/**********
						*qiaoyu add change ECO:0000xxx to evidence
						*
						*********/
						std::string sub = go.evidence_.substr(0,3);
						if(strcmp(sub.c_str(),"ECO") == 0) {
						
							std::string evidence_num = go.evidence_.substr(8,3);
							if(strcmp(evidence_num.c_str(),"269") == 0)  go.evidence_ = "EXP" ; 
							if(strcmp(evidence_num.c_str(),"318") == 0)  go.evidence_ = "IBA" ; 
							if(strcmp(evidence_num.c_str(),"319") == 0)  go.evidence_ = "IBD" ; 
							if(strcmp(evidence_num.c_str(),"305") == 0)  go.evidence_ = "IC" ; 
							if(strcmp(evidence_num.c_str(),"314") == 0)  go.evidence_ = "IDA" ; 
							if(strcmp(evidence_num.c_str(),"501") == 0)  go.evidence_ = "IEA" ; 
							if(strcmp(evidence_num.c_str(),"256") == 0)  go.evidence_ = "IEA" ; 
							if(strcmp(evidence_num.c_str(),"265") == 0)  go.evidence_ = "IEA" ; 
							if(strcmp(evidence_num.c_str(),"322") == 0)  go.evidence_ = "IEA" ; 
							if(strcmp(evidence_num.c_str(),"323") == 0)  go.evidence_ = "IEA" ; 
							if(strcmp(evidence_num.c_str(),"270") == 0)  go.evidence_ = "IEP" ; 
							if(strcmp(evidence_num.c_str(),"317") == 0)  go.evidence_ = "IGC" ; 
							if(strcmp(evidence_num.c_str(),"354") == 0)  go.evidence_ = "IGC" ; 
							if(strcmp(evidence_num.c_str(),"316") == 0)  go.evidence_ = "IGI" ; 
							if(strcmp(evidence_num.c_str(),"320") == 0)  go.evidence_ = "IKR" ; 
							if(strcmp(evidence_num.c_str(),"315") == 0)  go.evidence_ = "IMP" ; 
							//if(strcmp(evidence_num.c_str(),"320") == 0)  go.evidence_ = "IMR" ; 
							if(strcmp(evidence_num.c_str(),"353") == 0)  go.evidence_ = "IPI" ; 
							if(strcmp(evidence_num.c_str(),"321") == 0)  go.evidence_ = "IRD" ; 
							if(strcmp(evidence_num.c_str(),"247") == 0)  go.evidence_ = "ISA" ; 
							if(strcmp(evidence_num.c_str(),"255") == 0)  go.evidence_ = "ISM" ; 
							if(strcmp(evidence_num.c_str(),"266") == 0)  go.evidence_ = "ISO" ; 
							if(strcmp(evidence_num.c_str(),"250") == 0)  go.evidence_ = "ISS" ; 
							if(strcmp(evidence_num.c_str(),"031") == 0)  go.evidence_ = "ISS" ; 
							if(strcmp(evidence_num.c_str(),"303") == 0)  go.evidence_ = "NAS" ; 
							if(strcmp(evidence_num.c_str(),"307") == 0)  go.evidence_ = "ND" ; 
							if(strcmp(evidence_num.c_str(),"245") == 0)  go.evidence_ = "RCA" ; 
							if(strcmp(evidence_num.c_str(),"304") == 0)  go.evidence_ = "TAS" ;  
						}
					}
					go.id_ = atoi(str_id.data());
					add_go_term(go);
				}
			}
		}
		else if (v.first == "sequence") {
			string seq = v.second.data();
			for (auto u : seq) {
				AminoType amino = GetAminoType(string(1, u));
				if (amino != AminoType::NON)
					add_sequence_amino(amino);
			}
		}
	}
	SortGoTerm();
	return 1;
}

size_t ProteinSequence::ParseUniprotXml(const std::string& xml_file) {
	using namespace boost::property_tree;

	ptree rt_tree;
	try {
		read_xml(xml_file, rt_tree);
	} catch (const std::exception& e) {
		cerr << "Error: " << e.what() <<endl;
		return 0;
	}

	size_t success_cnt = 0;
	const ptree ptree_entry = rt_tree.get_child("uniprot.entry", EmptyPtree());
	if (ParseUniprotPtree(ptree_entry) > 0)
		++success_cnt;
	return success_cnt;
}

void  ProteinSequence::FilterGoByEvidence(const std::unordered_set<std::string>& evidences) {
	std::vector<GOType>  terms;
	std::unordered_set<std::string>::const_iterator it;
	for(size_t i = 0; i <go_terms_.size(); ++i) {

		std::string s = go_terms_.at(i).evidence_.substr(0,3);
		it = evidences.find(s);
		if(it != evidences.end()) terms.push_back(go_terms_.at(i));
		else{
			//evidence only two lettles "IC"
			std::string ss = go_terms_.at(i).evidence_.substr(0,2);
			it = evidences.find(ss);
			if(it != evidences.end()) terms.push_back(go_terms_.at(i));
		
		}
	}
	go_terms_ = terms;
}
void ProteinSequence::FilterGoCc() {
	std::vector<GOType>  terms;
	for(size_t i = 0; i <go_terms_.size(); ++i) {
	
		if(go_terms_.at(i).term_.at(0) != 'C')
			terms.push_back(go_terms_.at(i));
	}
	go_terms_ = terms;
}

void ProteinSequenceSet::FilterGoByEvidence(const std::unordered_set<std::string>& evidences) {
		for(size_t i = 0; i < protein_sequences_.size(); ++i) {
		
			 protein_sequences_.at(i).FilterGoByEvidence(evidences);
		}
		
}
void ProteinSequenceSet::FilterGoCc() {
	for(size_t i = 0; i < protein_sequences_.size(); ++i){
	
		protein_sequences_.at(i).FilterGoCc();
	}
}
	
void ProteinSequenceSet::FilterProteinSequenceBySpicies(const std::unordered_set<std::string>& spicies) {
	vector<ProteinSequence> ecopy;
	 std::unordered_set<std::string>::const_iterator it;
	 int x=0;
	 for(size_t i=0; i < protein_sequences_.size(); ++i) {
		it = spicies.find(protein_sequences_.at(i).species());
		if(it != spicies.end()) {
			ecopy.push_back(protein_sequences_.at(i));
		}
	 }
	 protein_sequences_ = ecopy;
}

size_t ProteinSequenceSet::ParseUniprotXml(const std::string& xml_file) {
	using namespace boost::property_tree;

	const string kXmlHead = "<entry ";
	const string kXmlTail = "</entry>";

	ifstream fin(xml_file);
	string line;
	string xml_str;
	bool b_begin = false;
	size_t success_cnt = 0;
	while (getline(fin, line)) {
		string tmp_str = trim_left_copy(line);
		if (strncmp(tmp_str.c_str(), kXmlHead.c_str(), kXmlHead.size()) == 0)
			b_begin = true;
		if (b_begin)
			xml_str += line +  "\n";
		if (strncmp(tmp_str.c_str(), kXmlTail.c_str(), kXmlTail.size()) == 0) {
			istringstream sin(xml_str);

			ptree rt_tree;
			try {
				read_xml(sin, rt_tree);
			} catch (const std::exception& e) {
				cerr << "Error: " << e.what() <<endl;
			}
			ProteinSequence protein_sequence;
			size_t tmp_ret = protein_sequence.ParseUniprotPtree(rt_tree.get_child("entry", EmptyPtree()));
			if (tmp_ret > 0) {
				protein_sequences_.push_back(protein_sequence);
				success_cnt += tmp_ret;
			}
			else
				cerr << "Warning: Parse ptree error!, name = " << protein_sequence.name() << endl;

			if (log_status() == LogStatus::FULL_LOG && (success_cnt & 4095) == 0)
				clog << "\rLoaded " << success_cnt << " successfully";

			xml_str.clear();
			b_begin = false;
		}
	}
	clog << endl;
	fin.close();

	if (success_cnt != protein_sequences().size())
	  cerr << "Error: success_cnt != protein_sequences().size()" << endl;

	if (log_status() != SILENT)
		clog << "Total loaded " << success_cnt << " instances successfully" << endl;
	return success_cnt;
}

size_t ProteinSequenceSet::ParseGoa(const std::string& goa_file) {
	const vector<string> kLeftSpecies = {"BACSU","PSEAE","STRPN","ARATH","YEAST","XENLA","HUMAN","MOUSE","RAT","DICDI","ECOLI"};
	ifstream fin(goa_file);
	string line;
	string go;
	string name;
	string name_last="";
	string accession;
	string last_accession;
	std::vector<ProteinSequence::GOType> go_terms;
	int first = 1;
	while (getline(fin, line)) {
		if (line[0] == '!')
			continue;
		name.clear();
		accession.clear();
		bool has_accession = false;
		if (boost::starts_with(line, "UniProtKB"))
			has_accession = true;
		string tmp_str = line;
		
		if (line == "" && (!last_accession.empty() || name_last.find("_") != string::npos)){
			ProteinSequence protein_sequence;
			if (!last_accession.empty())
				protein_sequence.add_accession(last_accession);
			protein_sequence.set_go_terms(go_terms);
			vector<string> tmpname2;
			boost::split( tmpname2, name_last, boost::is_any_of( "_" ), boost::token_compress_on );
			if (tmpname2.size() >= 2) {
				protein_sequence.set_name(tmpname2[0]);
				protein_sequence.set_species(tmpname2[1]);
			}
			protein_sequences_.push_back(protein_sequence);
			fin.close();
			return 1;
		}
		vector<string> v_str;
		boost::split( v_str, tmp_str, boost::is_any_of( "\t" ), boost::token_compress_off );
		
		vector<string> tmpname;
		boost::split( tmpname, v_str[10], boost::is_any_of( "|" ), boost::token_compress_on );
		if (has_accession)
			accession = v_str[1];
		else if (boost::starts_with(v_str[7], "UniProtKB:")){
			has_accession = true;
			accession = v_str[7].substr(v_str[7].find(":") + 1);
		}
		int cnt = 0;
		for (size_t i = 0; i < tmpname.size(); ++i) {
			bool is_name = false;
			for (const string& spice : kLeftSpecies) 
				if (tmpname[i].find("_" + spice) != string::npos) {
					is_name = true;
					break;
				}
			if (is_name) {
				name = tmpname[i];
				break;
			}
		}
		if (name.empty() && v_str[9].find("_") != string::npos) {
			size_t iter = string::npos;
			string tmp_spice;
			for (const string& spice : kLeftSpecies) 
				if ((iter = v_str[9].find("_" + spice)) != string::npos) {
					tmp_spice = spice;
					break;
				}
			if (!tmp_spice.empty()) {
				int end_index = iter;
				int start_index = end_index - 1;
				clog <<"Extract from " << v_str[9] << ", 9name = " ;
				while (start_index >= 0 && isalnum(v_str[9][start_index]))
					--start_index;
				name = v_str[9].substr(start_index + 1, end_index - start_index - 1) + "_" + tmp_spice;
				clog  << name << endl;
			}
		}

		if ((name != name_last && !name_last.empty()) || (accession != last_accession && !last_accession.empty())){
			
			if(!first && ((accession != last_accession && !last_accession.empty()) || name_last.find("_") != string::npos)){
				ProteinSequence protein_sequence;
				if (!last_accession.empty())
					protein_sequence.add_accession(last_accession);
				protein_sequence.set_go_terms(go_terms);
				vector<string> tmpname2;
				boost::split( tmpname2, name_last, boost::is_any_of( "_" ), boost::token_compress_on );
				if (tmpname2.size() >= 2) {
					protein_sequence.set_name(tmpname2[0]);
					protein_sequence.set_species(tmpname2[1]);
				}
				protein_sequences_.push_back(protein_sequence);
				if (log_status_ == FULL_LOG && protein_sequences_.size() % 10000 == 0)
					clog << "\r " << protein_sequences_.size() << " loaded";
			}
			go_terms.clear();
		}
		first = 0;
		
		ProteinSequence::GOType tempgo;
		tempgo.evidence_ = v_str[6];
		go = v_str[4];
		vector<string> tmpgoid;
		boost::split( tmpgoid, go, boost::is_any_of( ":" ), boost::token_compress_on );
		tempgo.id_ = atoi(tmpgoid[1].c_str());
		tempgo.term_ = v_str[8]+':'+v_str[9];
		go_terms.push_back(tempgo);
		name_last = name;
		last_accession = accession;
	}
	
	if (!last_accession.empty() || name_last.find("_") != string::npos) {
		ProteinSequence protein_sequence;
		if (!last_accession.empty())
			protein_sequence.add_accession(last_accession);
		protein_sequence.set_go_terms(go_terms);
		vector<string> tmpname2;
		boost::split( tmpname2, name_last, boost::is_any_of( "_" ), boost::token_compress_on );
		if (tmpname2.size() >= 2) {
			protein_sequence.set_name(tmpname2[0]);
			protein_sequence.set_species(tmpname2[1]);
		}
		protein_sequences_.push_back(protein_sequence);
	}
	fin.close();
	return 1;
}

size_t ProteinSequenceSet::ParseDat(const std::string& goa_file) {
	ifstream fin(goa_file);
	string str,tag;
	ProteinSequence ps;
	size_t success_cnt = 0;
	while(getline(fin,str)){
		stringstream ssin;
		ssin.clear();
		ssin<<str;
		ssin>>tag;
		
		//name_spices reviewed
		if(tag == "ID"){
			string id,name,spc;
			ssin>>id;
			if(id != ""){
				size_t pos = id.find("_") ;
				name=id.substr(0,pos);
				spc=id.substr(pos+1);
			}
			ps.set_name(name);
			ps.set_species(spc);
		}
		//accessions 
		else if(tag == "AC"){
			string tmp;
			while(ssin>>tmp){
				tmp = tmp.substr(0,tmp.length()-1);
				ps.add_accession(tmp);
			}
		}
		//pubmed=id;
		else if(tag == "RX"){
			string pub;
			while(ssin >> pub){
				string flag=pub.substr(0,6);
				if(flag == "PubMed"){
					int pub_id;
					stringstream ss;
					ss<<pub.substr(7,pub.length()-8);
					ss>>pub_id;
					ps.add_ref_pmid(pub_id);
					ss.clear();
					
				}
			}
		}
		//GO terms
		else if(tag == "DR"){
			string str_go;
			ssin>>str_go;
			
			if(str_go == "GO;"){
				ProteinSequence::GOType go_terms;
				string s_id,s_term,s_evidence;
				ssin>>s_id;
				
				 stringstream ss;
				 ss<<s_id.substr(3,s_id.length()-4);
				 ss>>go_terms.id_;
				 ss.clear();
				
				 string tmp;
				getline(ssin,tmp,';');
				go_terms.term_=tmp.substr(1,tmp.length()-1);
				
				ssin>>s_evidence;
				int pos = s_evidence.find(":");
				go_terms.evidence_=s_evidence.substr(0,pos);
				ps.add_go_term(go_terms);
			}
			
		}
		//set sequence
		else if(tag == "SQ"){
			string s_sq,tmp;
			while(getline(fin,str)){
				if(str != "//"){
					for (auto u : str) {
						AminoType amino = GetAminoType(string(1, u));
						if (amino != AminoType::NON)
							ps.add_sequence_amino(amino);
					}
				}
				else{
					
					protein_sequences_.push_back(ps);
					ps.clear();
					success_cnt =protein_sequences_.size() ;
					if(success_cnt)
						clog << "\rLoaded \t" << success_cnt << "\tsuccessfully";
					break;
				}
			}
			
		}
		else continue;

	}
	fin.close();
	clog<<endl;
	return 1;
}

void ProteinSequenceSet::FilterNotIndexedProtein() {
	vector<ProteinSequence> indexed_proteins;
	for (const ProteinSequence& protein : protein_sequences())
		if (!protein.go_terms().empty())
			indexed_proteins.push_back(protein);
	
	protein_sequences_ = indexed_proteins;
}

void ProteinSequenceSet::ReserveByNameSpice(const ProteinSequenceSet& protein_set) {
	unordered_set<string> substract_proteins;
	for (const ProteinSequence& protein : protein_set.protein_sequences())
		substract_proteins.insert(protein.name() + "_" + protein.species());
	
	vector<ProteinSequence> reserved_proteins;
	for (const ProteinSequence& protein : protein_sequences())
		if (substract_proteins.find(protein.name() + "_" + protein.species()) != substract_proteins.end())
			reserved_proteins.push_back(protein);
	protein_sequences_ = reserved_proteins;
}

ProteinSequenceSet ProteinSequenceSet::SubtractByNameSpice(const ProteinSequenceSet& protein_set) const {
	unordered_set<string> substract_proteins;
	for (const ProteinSequence& protein : protein_set.protein_sequences())
		substract_proteins.insert(protein.name() + "_" + protein.species());
	//clog << "substract_proteins.size = " << substract_proteins.size() << " protein_set.size = " << protein_set.protein_sequences().size() << endl;
	ProteinSequenceSet ret_protein_set;
	for (const ProteinSequence& protein : protein_sequences())
		if (substract_proteins.find(protein.name() + "_" + protein.species()) == substract_proteins.end())
			ret_protein_set.add_protein_sequence(protein);
	return ret_protein_set;
}

ProteinSequenceSet ProteinSequenceSet::SubtractByAccession(const ProteinSequenceSet& protein_set) const {
	unordered_set<string> substract_proteins;
	for (const ProteinSequence& protein : protein_set.protein_sequences()) {
		for (const string& accession : protein.accessions())
			substract_proteins.insert(accession);
	}
	//clog << "substract_proteins.size = " << substract_proteins.size() << " protein_set.size = " << protein_set.protein_sequences().size() << endl;
	ProteinSequenceSet ret_protein_set;
	for (const ProteinSequence& protein : protein_sequences()) {
		bool b_add = true;
		for (const string &accession : protein.accessions())
			if (substract_proteins.find(accession) != substract_proteins.end()) {
				b_add = false;
				break;
			}
		if (b_add)
			ret_protein_set.add_protein_sequence(protein);
	}
	return ret_protein_set;
}

void ProteinSequenceSet::Save(const std::string& file_name) const {
	ofstream fout(file_name);
	boost::archive::binary_oarchive oa(fout);
	oa << *this;
	fout.close();
}

void ProteinSequenceSet::SaveToFasta(const std::string& file_name) const{
        ofstream fout(file_name);
	for(size_t i = 0; i < protein_sequences_.size(); ++i) {
		ProteinSequence item = protein_sequences_[i];
		fout<<">";
		fout<<"sp|";
		std::vector<std::string> vacc = item. accessions();
		if (vacc.empty())
			fout << "000000";
		else
			fout<<vacc.at(0);
		fout<<"|";
		fout<<item.name()<<"_"<<item.species()<<endl;
		std::vector<AminoType> ami = item.sequence();
		for(size_t j = 0; j < ami.size(); ++j) {
	  		fout<<Get1LetterAminoName(ami[j]);
		}
		fout<<endl;
	}
	clog	<<"Total save "<<protein_sequences_.size()<<" fasta proteins!"<<endl;
        fout.close();
}

void ProteinSequenceSet::SaveToString(const std::string& file_name) const{
	 ofstream fout(file_name);
	 for(size_t i = 0; i < protein_sequences_.size(); ++i) {
	 
		 fout<<protein_sequences_.at(i).ToString();
		 if (i % 10000 == 0)
			clog << "\rSaved " << i +1<< " successfully";
	}
	clog << "Total saved " << protein_sequences().size() << " proteins";
	clog <<endl;
}

size_t ProteinSequenceSet::Load(const std::string& file_name) {
	protein_sequences_.clear();

	ifstream fin(file_name);
	boost::archive::binary_iarchive ia(fin);
	ia >> *this;
	if (log_status() != LogStatus::SILENT)
		clog << "Total load " << protein_sequences_.size() << " protein sequences!" << endl;
	fin.close();
	return protein_sequences_.size();
}
