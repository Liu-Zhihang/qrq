// Added only to an isolated source copy. Shared static DFS valid-rank layout.
std::vector<std::size_t> matched_lo_, matched_hi_, matched_rank_, matched_ids_;

void matched_layout_visit(std::size_t id) {
    matched_lo_[id]=matched_ids_.size();
    for(const auto& e:nodes_[id].entries) if(e.valid) {
        if(e.child>=0) matched_layout_visit(static_cast<std::size_t>(e.child));
        else { matched_rank_[e.item]=matched_ids_.size(); matched_ids_.push_back(e.item); }
    }
    matched_hi_[id]=matched_ids_.size();
}

void matched_ensure_layout() {
    if(!matched_ids_.empty()) return;
    matched_lo_.resize(nodes_.size()); matched_hi_.resize(nodes_.size());
    matched_rank_.resize(segments_.size()); matched_layout_visit(root_);
}

py::dict matched_layout() {
    matched_ensure_layout();
    return py::dict("rank_to_id"_a=matched_ids_,"id_to_rank"_a=matched_rank_,
        "nodes"_a=nodes_.size(),"root_height"_a=nodes_[root_].height,
        "n"_a=segments_.size(),"rank_interval_words"_a=2*nodes_.size(),
        "permutation_words"_a=2*segments_.size());
}

struct MatchedPlan {
    std::uint64_t node_reads=0,slots=0,geometry=0;
    std::vector<std::pair<std::size_t,std::size_t>> ranges;
};

void matched_classical_visit(std::size_t id,const Box3& query,bool blocks,MatchedPlan& plan) {
    ++plan.node_reads; const auto& node=nodes_[id];
    if(!intersects(query,node.box)) return;
    if(blocks && contains(query,node.box)) {
        plan.ranges.emplace_back(matched_lo_[id],matched_hi_[id]); return;
    }
    for(const auto& e:node.entries) {
        ++plan.slots;
        if(!e.valid || !intersects(e.box,query)) continue;
        if(e.child>=0) matched_classical_visit(e.child,query,blocks,plan);
        else {
            ++plan.geometry;
            if(segment_intersects(segments_[e.item],query))
                plan.ranges.emplace_back(matched_rank_[e.item],matched_rank_[e.item]+1);
        }
    }
}

py::dict matched_query(double xmin,double xmax,double ymin,double ymax,double tmin,double tmax) {
    matched_ensure_layout(); const auto query=make_query(xmin,xmax,ymin,ymax,tmin,tmax);
    SamplingDiscovery discovery;
    sampling_discover(query,std::vector<std::size_t>{root_},0,discovery);
    std::vector<std::pair<std::size_t,std::size_t>> candidate;
    for(auto id:discovery.frontier) candidate.emplace_back(matched_lo_[id],matched_hi_[id]);
    MatchedPlan block,enumeration;
    matched_classical_visit(root_,query,true,block);
    matched_classical_visit(root_,query,false,enumeration);
    auto pack=[](const MatchedPlan& p) { return py::dict("node_reads"_a=p.node_reads,
        "slots"_a=p.slots,"geometry"_a=p.geometry,"ranges"_a=p.ranges); };
    return py::dict("candidate_ranges"_a=candidate,"prefix_nodes"_a=discovery.node_reads,
        "prefix_slots"_a=discovery.scan_slots,"prefix_geometry"_a=discovery.geometry_tests,
        "frontier_size"_a=discovery.frontier.size(),"height"_a=discovery.height,
        "depth"_a=discovery.depth,"witness_kind"_a=discovery.kind,
        "block"_a=pack(block),"enumeration"_a=pack(enumeration));
}
