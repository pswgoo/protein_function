#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>

class GOTerm {
public:
	std::string ToString() const;

	int id() const { return id_; }
	void set_id(const int& id) { id_ = id; }
	
	const std::vector<int>& alt_ids() const { return alt_ids_; }
	void set_alt_ids(const std::vector<int>& ids) { alt_ids_ = ids; }
	void add_alt_id(int id) { alt_ids_.push_back(id); }

	const std::vector<int>& fathers() const { return fathers_; }
	void set_fathers(const std::vector<int>& fathers) { fathers_ = fathers;  SortFather(); }

	const std::string& name() const { return name_; }
	void set_name(const std::string& name) { name_ = name; }

	const std::string& type() const { return type_; }
	void set_type(const std::string& type) { type_ = type; }

	std::vector<int> FindAncestor(const int& node);
	void add_father(const int father) { fathers_.push_back(father); SortFather();}
private:
	friend boost::serialization::access;
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & id_  & alt_ids_ & name_ & fathers_ & type_;
	}

	void SortFather() {
		std::sort(fathers_.begin(), fathers_.end());
	}

private:
	int id_;
	std::vector<int> alt_ids_;
	std::string name_;
	std::vector<int> fathers_;
	std::string type_;	//namespace
};

class GOTermSet {
public:
	std::vector<int> FindAncestors(const std::vector<int>& go_term_ids) const;
	
	bool HasKey(int go_id) const { return go_term_index_.count(go_id) > 0; }

	const GOTerm& QueryGOTerm(int go_id) const {
		return go_terms().at(go_term_index_.at(go_id));
	}

	std::size_t ParseGo(const std::string& go_file);

	void Save(const std::string& file_name) const;

	std::size_t Load(const std::string& file_name);

	int update_date() const { return update_date_; }
	void set_update_date(int update_date) { update_date_ = update_date; }

	const std::vector<GOTerm>& go_terms() const { return go_terms_; }
	std::size_t add_go_term(const GOTerm& go_term) {
		if (go_term_index_.count(go_term.id()) > 0)
			return 0;
		go_terms_.push_back(go_term);
		go_term_index_[go_term.id()] = go_terms_.size() - 1;
		for (int alt_id : go_term.alt_ids()) {
			if (go_term_index_.count(alt_id) > 0)
				std::clog << "Warning: " << alt_id << " already exists" << std::endl;
			go_term_index_[alt_id] = go_terms_.size() - 1;
		}
		return 1;
	}

private:
	friend boost::serialization::access;
	template<typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & update_date_ & go_term_index_ & go_terms_;
	}

private:
	/*!@brief yyyymmdd*/
	int update_date_;
	std::unordered_map<int, std::size_t> go_term_index_;
	std::vector<GOTerm> go_terms_;
};
