friend class boost::serialization::access;


// When the class Archive corresponds to an output archive, the
// & operator is defined similar to <<.  Likewise, when the class Archive
// is a type of input archive the & operator is defined similar to >>.
template<class Archive>
inline void serialize(Archive & ar, const unsigned int version)
{
    int r = this->rows();     // makes sense for saving only
    int c = this->cols();

    ar & boost::serialization::make_nvp("rows",r);
    ar & boost::serialization::make_nvp("cols",c);

    this->resize(r,c);        // makes sense for loading only

    for(int j = 0; j < c; j++)
	    for(int i = 0; i < r; i++)
            ar & boost::serialization::make_nvp("data",this->operator()(i,j));
}

