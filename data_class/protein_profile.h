#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>

class ProteinProfile {

public:
	std::string ToString() const;

	/**
	* @param ptree : uniprot database xml ptree structure, the ptree only contains a protein sequence
	* @return 1 if load successfully, else return 0
	*/
	size_t ParseUniprotPtree(const boost::property_tree::ptree& ptree);

	/**
	* @param xml_file : uniprot database xml file name, the xml file only contains a protein sequence
	* @return 1 if load successfully, else return 0
	*/
	size_t ParseUniprotXml(const std::string& xml_file);

	const std::string& name() const { return name_; }
	void set_name(const std::string& name) { name_ = boost::to_upper_copy(name); }

	const std::string& species() const { return species_; }
	void set_species(const std::string& species) { species_ = boost::to_upper_copy(species); }

	const std::vector<std::string>& accessions() const { return accessions_; }
	void set_accessions(const std::vector<std::string>& accessions) {
		for (auto &u : accessions)
			accessions_.push_back(boost::to_upper_copy(u));
	}
	void add_accession(const std::string& accession) { accessions_.push_back(accession); }

	const std::vector<int>& ref_pmids() const { return ref_pmids_; }
	void set_ref_pmids(const std::vector<int>& ref_pmids) { ref_pmids_ = ref_pmids; }
	void add_ref_pmid(int pmid) { ref_pmids_.push_back(pmid); }

	void Clear() { name_ = ""; species_ = ""; accessions_.clear(); ref_pmids_.clear(); }
private:
	friend boost::serialization::access;
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & name_  & species_ & accessions_ & ref_pmids_;
	}

private:
	std::string name_;
	std::string species_;
	std::vector<std::string> accessions_;
	std::vector<int> ref_pmids_;
};

class ProteinProfileSet {
public:
	ProteinProfileSet() = default;
	ProteinProfileSet(const std::string & file_name) { Load(file_name); }

	/**
	* @param xml_file : uniprot database xml file name, the xml file contains all protein sequence information
	* @return the number of protein sequences loaded successfully
	*/
	std::size_t ParseUniprotXml(const std::string& xml_file);

	void Save(const std::string& file_name) const;
	std::size_t Load(const std::string& file_name);

	bool Has(const std::string &name) {
		if (protein_indices_.count(name) > 0)
			return true;
		else
			return false;
	}

	void ProteinProfileSet::Print(const std::string& file_name) const {
		std::ofstream fout(file_name);
		for (size_t i = 0; i < protein_profiles_.size(); ++i) {
			fout << protein_profiles_.at(i).ToString();
			if (i % 10000 == 0)
				std::clog << "\rSaved " << i + 1 << " successfully";
		}
		std::clog << "Total saved " << protein_profiles_.size() << " proteins" << std::endl;
	}

private:
	friend boost::serialization::access;
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & protein_profiles_ & protein_indices_;
	}

private:
	std::vector<ProteinProfile> protein_profiles_;
	std::unordered_map<std::string, std::size_t> protein_indices_;
};
