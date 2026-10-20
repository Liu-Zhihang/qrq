#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

using Vec3 = std::array<double, 3>;

struct Box3 {
    Vec3 lower{};
    Vec3 upper{};
};

struct Segment3 {
    Vec3 start{};
    Vec3 end{};
    Box3 box{};
    std::int64_t trajectory{};
};

struct TrajectoryQueryResult3D {
    std::uint64_t result_count{};
    std::uint64_t classical_io{};
    std::uint64_t quantum_io{};
    double probability_aware_io{};

    double model_speedup() const {
        return quantum_io == 0
            ? std::numeric_limits<double>::infinity()
            : static_cast<double>(classical_io) / static_cast<double>(quantum_io);
    }
};

struct TrajectoryBenchmarkResult3D {
    std::size_t query_count{};
    double average_answers{};
    double average_classical_io{};
    double average_quantum_io{};
    double average_probability_aware_io{};

    double model_speedup() const {
        return average_quantum_io == 0.0
            ? std::numeric_limits<double>::infinity()
            : average_classical_io / average_quantum_io;
    }
};

bool intersects(const Box3& a, const Box3& b) {
    for (int axis = 0; axis < 3; ++axis) {
        if (a.upper[axis] < b.lower[axis] || a.lower[axis] > b.upper[axis]) {
            return false;
        }
    }
    return true;
}

bool contains(const Box3& outer, const Box3& inner) {
    for (int axis = 0; axis < 3; ++axis) {
        if (outer.lower[axis] > inner.lower[axis] || outer.upper[axis] < inner.upper[axis]) {
            return false;
        }
    }
    return true;
}

bool segment_intersects(const Segment3& segment, const Box3& query) {
    double enter = 0.0;
    double leave = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double direction = segment.end[axis] - segment.start[axis];
        if (std::abs(direction) <= std::numeric_limits<double>::epsilon()) {
            if (segment.start[axis] < query.lower[axis] ||
                segment.start[axis] > query.upper[axis]) {
                return false;
            }
            continue;
        }
        double first = (query.lower[axis] - segment.start[axis]) / direction;
        double second = (query.upper[axis] - segment.start[axis]) / direction;
        if (first > second) {
            std::swap(first, second);
        }
        enter = std::max(enter, first);
        leave = std::min(leave, second);
        if (enter > leave) {
            return false;
        }
    }
    return true;
}

Vec3 center(const Box3& box) {
    return Vec3{
        (box.lower[0] + box.upper[0]) * 0.5,
        (box.lower[1] + box.upper[1]) * 0.5,
        (box.lower[2] + box.upper[2]) * 0.5
    };
}

Box3 union_boxes(const Box3& a, const Box3& b) {
    Box3 result;
    for (int axis = 0; axis < 3; ++axis) {
        result.lower[axis] = std::min(a.lower[axis], b.lower[axis]);
        result.upper[axis] = std::max(a.upper[axis], b.upper[axis]);
    }
    return result;
}

enum class TreeKind { PackedRTree, TBTree, KDTree };

TreeKind parse_kind(const std::string& value) {
    if (value == "packed_rtree") return TreeKind::PackedRTree;
    if (value == "tb_tree") return TreeKind::TBTree;
    if (value == "kd_tree") return TreeKind::KDTree;
    throw std::invalid_argument("tree_type must be packed_rtree, tb_tree, or kd_tree");
}

}  // namespace

class QuantumTrajectoryTree3D {
public:
    QuantumTrajectoryTree3D(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& starts,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& ends,
        const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& trajectories,
        int branching_factor,
        const std::string& tree_type
    ) : branching_factor_(branching_factor), kind_(parse_kind(tree_type)), tree_type_(tree_type) {
        if (branching_factor_ < 2 ||
            (branching_factor_ & (branching_factor_ - 1)) != 0) {
            throw std::invalid_argument("branching_factor must be a power of two and at least 2");
        }
        read_segments(starts, ends, trajectories);
        build();
    }

    TrajectoryQueryResult3D query(
        double xmin, double xmax,
        double ymin, double ymax,
        double tmin, double tmax,
        std::uint64_t seed
    ) const {
        const Box3 query_box = make_query(xmin, xmax, ymin, ymax, tmin, tmax);
        std::mt19937_64 random(seed);
        return query_with_rng(query_box, random);
    }

    std::vector<std::size_t> query_ids(
        double xmin, double xmax,
        double ymin, double ymax,
        double tmin, double tmax
    ) const {
        const Box3 query_box = make_query(xmin, xmax, ymin, ymax, tmin, tmax);
        std::vector<std::size_t> result;
        collect_ids(root_, query_box, result);
        std::sort(result.begin(), result.end());
        return result;
    }

    py::dict count_audit_query(
        double xmin, double xmax,
        double ymin, double ymax,
        double tmin, double tmax
    ) const {
        const Box3 query_box = make_query(xmin, xmax, ymin, ymax, tmin, tmax);
        CountAuditStats stats;
        const std::uint64_t count = classical_count_with_containment(root_, query_box, stats);
        return py::dict(
            "result_count"_a = count,
            "count_node_reads"_a = stats.node_reads,
            "count_entry_slot_reads"_a = stats.entry_slot_reads,
            "count_contained_shortcuts"_a = stats.contained_shortcuts,
            "count_records_via_shortcut"_a = stats.records_via_shortcut,
            "count_exact_segment_tests"_a = stats.exact_segment_tests,
            "count_valid_entries_examined"_a = stats.valid_entries_examined
        );
    }

    py::dict sample_query(
        double xmin, double xmax,
        double ymin, double ymax,
        double tmin, double tmax,
        const std::string& proposal_profile,
        std::uint64_t seed,
        std::size_t samples,
        std::uint64_t max_trials,
        bool audit_support,
        std::uint64_t support_limit
    ) const {
        if (proposal_profile != "padded" && proposal_profile != "valid") {
            throw std::invalid_argument("proposal_profile must be padded or valid");
        }
        if (samples == 0 || samples > 1000000) {
            throw std::invalid_argument("samples must be between 1 and 1000000");
        }
        for (double bound : {xmin, xmax, ymin, ymax, tmin, tmax}) {
            if (std::isnan(bound)) {
                throw std::invalid_argument("sample_query bounds must not contain NaN");
            }
        }
        const Box3 box = make_query(xmin, xmax, ymin, ymax, tmin, tmax);
        SamplingDiscovery discovery;
        sampling_discover(box, std::vector<std::size_t>{root_}, 0, discovery);
        const std::uint64_t block = static_cast<std::uint64_t>(branching_factor_);
        const std::uint64_t derived_nodes = discovery.scan_slots / block +
            (discovery.scan_slots % block != 0);
        if (discovery.node_reads != derived_nodes) {
            throw std::logic_error("sampling discovery is not a physical-slot prefix");
        }

        const bool valid_profile = proposal_profile == "valid";
        std::vector<std::size_t> cumulative_weights;
        std::size_t records = 0;
        std::uint64_t preparation_weight_reads = 0;
        if (valid_profile) {
            cumulative_weights.reserve(discovery.frontier.size());
            for (std::size_t node_id : discovery.frontier) {
                ++preparation_weight_reads;
                records += nodes_[node_id].weight;
                cumulative_weights.push_back(records);
            }
        }
        const double padded_domain = static_cast<double>(discovery.frontier.size()) *
            std::pow(static_cast<double>(branching_factor_), discovery.height);
        const double proposal_domain = valid_profile
            ? static_cast<double>(records) : padded_domain;
        SamplingLocalStats local;
        std::mt19937_64 random(seed);
        std::vector<std::size_t> sampled_ids;
        std::vector<std::uint64_t> trials_per_sample, node_reads_per_sample, weight_reads_per_sample;
        sampled_ids.reserve(std::min<std::size_t>(samples, 1024));
        std::uint64_t trials = 0, pending_trials = 0;
        std::uint64_t dummy_rejections = 0, geometry_rejections = 0, geometry_tests = 0;
        std::uint64_t completed_node_reads = 0, completed_weight_reads = 0;
        if (!discovery.frontier.empty()) {
            std::uniform_int_distribution<std::size_t> choose_root(0, discovery.frontier.size() - 1);
            std::uniform_int_distribution<int> choose_slot(0, branching_factor_ - 1);
            std::uniform_int_distribution<std::size_t> choose_rank(0, records > 0 ? records - 1 : 0);
            while (sampled_ids.size() < samples && trials < max_trials) {
                ++trials;
                ++pending_trials;
                const std::size_t item = valid_profile
                    ? sampling_route_rank(discovery.frontier, cumulative_weights, choose_rank(random), local)
                    : sampling_route_padded(
                        discovery.frontier[choose_root(random)], discovery.height,
                        [&]() { return static_cast<std::size_t>(choose_slot(random)); }, local
                    );
                if (item == sampling_dummy()) {
                    ++dummy_rejections;
                    continue;
                }
                ++geometry_tests;
                if (!segment_intersects(segments_[item], box)) {
                    ++geometry_rejections;
                    continue;
                }
                sampled_ids.push_back(item);
                trials_per_sample.push_back(pending_trials);
                node_reads_per_sample.push_back(local.node_reads - completed_node_reads);
                weight_reads_per_sample.push_back(local.weight_reads - completed_weight_reads);
                pending_trials = 0;
                completed_node_reads = local.node_reads;
                completed_weight_reads = local.weight_reads;
            }
        }
        const std::string status = discovery.frontier.empty() ? "empty"
            : sampled_ids.size() == samples ? "ok" : "budget_exhausted";
        py::dict result(
            "status"_a = status,
            "proposal_profile"_a = proposal_profile,
            "requested_samples"_a = samples,
            "completed_samples"_a = sampled_ids.size(),
            "sampled_ids"_a = sampled_ids,
            "trials_per_sample"_a = trials_per_sample,
            "total_trials"_a = trials,
            "unfinished_sample_trials"_a = pending_trials,
            "global_scan_slots"_a = discovery.scan_slots,
            "global_node_reads"_a = discovery.node_reads,
            "global_geometry_tests"_a = discovery.geometry_tests,
            "frontier_nodes"_a = discovery.frontier.size(),
            "witness_height"_a = discovery.height,
            "witness_depth"_a = discovery.depth,
            "witness_kind"_a = discovery.kind,
            "frontier_records"_a = valid_profile ? py::cast(records) : py::none(),
            "preparation_weight_reads"_a = preparation_weight_reads,
            "local_node_reads"_a = local.node_reads,
            "local_weight_reads"_a = local.weight_reads,
            "local_node_reads_per_sample"_a = node_reads_per_sample,
            "local_weight_reads_per_sample"_a = weight_reads_per_sample,
            "unfinished_local_node_reads"_a = local.node_reads - completed_node_reads,
            "unfinished_local_weight_reads"_a = local.weight_reads - completed_weight_reads,
            "fixed_model_local_node_reads"_a = static_cast<double>(discovery.height) * static_cast<double>(trials),
            "dummy_rejections"_a = dummy_rejections,
            "geometry_rejections"_a = geometry_rejections,
            "geometry_tests"_a = geometry_tests,
            "padded_domain"_a = padded_domain,
            "proposal_domain"_a = proposal_domain
        );
        const py::dict support = sampling_audit_support(
            box, discovery, cumulative_weights, records, valid_profile,
            audit_support, support_limit, proposal_domain
        );
        for (auto entry : support) result[entry.first] = entry.second;
        return result;
    }

    py::dict audit_query(
        double xmin, double xmax,
        double ymin, double ymax,
        double tmin, double tmax,
        bool verify_coverage
    ) const {
        const Box3 query_box = make_query(xmin, xmax, ymin, ymax, tmin, tmax);
        std::uint64_t classical_io = 0;
        const std::uint64_t count = classical_count(root_, query_box, classical_io);
        AuditFrontier audit;
        audit_frontier(
            query_box, std::vector<std::size_t>{root_}, 0, verify_coverage, audit
        );
        if ((count > 0) != (audit.witness_height > 0)) {
            throw std::logic_error("exact answers and audit witness disagree");
        }

        const double scans = static_cast<double>(audit.scan_slots);
        double valid_expected_io = scans;
        double legacy_expected_io = scans;
        double padded_expected_io = scans;
        double padded_domain = 0.0;
        double valid_success_probability = 0.0;
        double padded_success_probability = 0.0;
        if (count > 0) {
            const double answers = static_cast<double>(count);
            const double records = static_cast<double>(audit.frontier_records);
            const double witness_height = static_cast<double>(audit.witness_height);
            padded_domain = static_cast<double>(audit.frontier_nodes) * std::pow(
                static_cast<double>(branching_factor_), witness_height
            );
            valid_success_probability = answers / records;
            padded_success_probability = answers / padded_domain;
            valid_expected_io += witness_height * records / answers;
            // Preserve the released draw <= k success boundary as a separate
            // expectation; no random draws or historical counters are changed.
            legacy_expected_io += witness_height * records / std::min(records, answers + 1.0);
            // This is a counterfactual complete B-ary path domain, not a claim
            // that a padded-state preparation circuit has been implemented.
            padded_expected_io += witness_height * padded_domain / answers;
        }

        py::object candidate_covers_answers = py::none();
        if (verify_coverage) {
            std::vector<std::size_t> candidates;
            candidates.reserve(audit.frontier_records);
            for (std::size_t node_id : audit.witness_frontier) {
                collect_subtree_ids(node_id, candidates);
            }
            std::vector<std::size_t> answers;
            collect_ids(root_, query_box, answers);
            std::sort(candidates.begin(), candidates.end());
            std::sort(answers.begin(), answers.end());
            candidate_covers_answers = py::bool_(std::includes(
                candidates.begin(), candidates.end(), answers.begin(), answers.end()
            ));
        }

        return py::dict(
            "result_count"_a = count,
            "classical_io"_a = classical_io,
            "scan_slots"_a = audit.scan_slots,
            "frontier_nodes"_a = audit.frontier_nodes,
            "frontier_records"_a = audit.frontier_records,
            "witness_height"_a = audit.witness_height,
            "witness_kind"_a = audit.witness_kind,
            "witness_depth"_a = audit.witness_depth,
            "valid_expected_io"_a = valid_expected_io,
            "legacy_expected_io"_a = legacy_expected_io,
            "padded_domain"_a = padded_domain,
            "padded_expected_io"_a = padded_expected_io,
            "valid_success_probability"_a = valid_success_probability,
            "padded_success_probability"_a = padded_success_probability,
            "candidate_covers_answers"_a = candidate_covers_answers,
            "tree_type"_a = tree_type_,
            "branching_factor"_a = branching_factor_,
            "tree_height"_a = height(),
            "truth_source"_a = "offline_classical_exact_count"
        );
    }

    TrajectoryBenchmarkResult3D benchmark(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& queries,
        std::uint64_t seed
    ) const {
        const py::buffer_info info = queries.request();
        if (info.ndim != 2 || info.shape[1] != 6 || info.shape[0] == 0) {
            throw std::invalid_argument("queries must have shape (Q, 6) and be non-empty");
        }
        const auto* values = static_cast<const double*>(info.ptr);
        std::mt19937_64 random(seed);
        long double answers = 0.0L;
        long double classical = 0.0L;
        long double quantum = 0.0L;
        long double probability = 0.0L;
        for (py::ssize_t row = 0; row < info.shape[0]; ++row) {
            const double* q = values + row * 6;
            const TrajectoryQueryResult3D result = query_with_rng(
                make_query(q[0], q[1], q[2], q[3], q[4], q[5]), random
            );
            answers += result.result_count;
            classical += result.classical_io;
            quantum += result.quantum_io;
            probability += result.probability_aware_io;
        }
        const long double count = static_cast<long double>(info.shape[0]);
        return TrajectoryBenchmarkResult3D{
            static_cast<std::size_t>(info.shape[0]),
            static_cast<double>(answers / count),
            static_cast<double>(classical / count),
            static_cast<double>(quantum / count),
            static_cast<double>(probability / count)
        };
    }

    #include "qrq_matched_audit.inc"

    std::size_t size() const { return segments_.size(); }
    std::size_t node_count() const { return nodes_.size(); }
    int height() const { return nodes_.at(root_).height; }
    int branching_factor() const { return branching_factor_; }
    const std::string& tree_type() const { return tree_type_; }

private:
    struct Entry {
        Box3 box{};
        std::ptrdiff_t child{-1};
        std::ptrdiff_t item{-1};
        bool valid{false};
    };

    struct Node {
        std::vector<Entry> entries;
        Box3 box{};
        std::size_t weight{};
        int height{1};
        bool leaf{false};
    };

    struct CountAuditStats {
        std::uint64_t node_reads{};
        std::uint64_t entry_slot_reads{};
        std::uint64_t contained_shortcuts{};
        std::uint64_t records_via_shortcut{};
        std::uint64_t exact_segment_tests{};
        std::uint64_t valid_entries_examined{};
    };

    struct AuditFrontier {
        std::uint64_t scan_slots{};
        std::size_t frontier_nodes{};
        std::size_t frontier_records{};
        int witness_height{};
        int witness_depth{-1};  // Root depth is zero; -1 denotes no witness.
        std::string witness_kind{"none"};
        // Only retained for the opt-in exhaustive coverage check.
        std::vector<std::size_t> witness_frontier;
    };

    struct SamplingDiscovery {
        std::uint64_t scan_slots{};
        std::uint64_t node_reads{};
        std::uint64_t geometry_tests{};
        std::vector<std::size_t> frontier;
        int height{};
        int depth{-1};
        std::string kind{"none"};
    };

    struct SamplingLocalStats {
        std::uint64_t node_reads{};
        std::uint64_t weight_reads{};
    };

    std::vector<Segment3> segments_;
    int branching_factor_;
    TreeKind kind_;
    std::string tree_type_;
    std::vector<Node> nodes_;
    std::size_t root_{};

    static Box3 make_query(
        double xmin, double xmax, double ymin, double ymax, double tmin, double tmax
    ) {
        if (xmin > xmax || ymin > ymax || tmin > tmax) {
            throw std::invalid_argument("query lower bounds must not exceed upper bounds");
        }
        return Box3{{xmin, ymin, tmin}, {xmax, ymax, tmax}};
    }

    void read_segments(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& starts,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& ends,
        const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& trajectories
    ) {
        const py::buffer_info start_info = starts.request();
        const py::buffer_info end_info = ends.request();
        const py::buffer_info trajectory_info = trajectories.request();
        if (start_info.ndim != 2 || start_info.shape[1] != 3 ||
            end_info.ndim != 2 || end_info.shape[1] != 3 ||
            trajectory_info.ndim != 1 || start_info.shape[0] == 0 ||
            end_info.shape[0] != start_info.shape[0] ||
            trajectory_info.shape[0] != start_info.shape[0]) {
            throw std::invalid_argument("starts/ends must be (N,3), trajectories must be (N,), N>0");
        }
        const auto* start_values = static_cast<const double*>(start_info.ptr);
        const auto* end_values = static_cast<const double*>(end_info.ptr);
        const auto* trajectory_values = static_cast<const std::int64_t*>(trajectory_info.ptr);
        segments_.reserve(static_cast<std::size_t>(start_info.shape[0]));
        for (py::ssize_t row = 0; row < start_info.shape[0]; ++row) {
            Segment3 segment;
            for (int axis = 0; axis < 3; ++axis) {
                segment.start[axis] = start_values[row * 3 + axis];
                segment.end[axis] = end_values[row * 3 + axis];
                if (!std::isfinite(segment.start[axis]) || !std::isfinite(segment.end[axis])) {
                    throw std::invalid_argument("segment coordinates must be finite");
                }
                segment.box.lower[axis] = std::min(segment.start[axis], segment.end[axis]);
                segment.box.upper[axis] = std::max(segment.start[axis], segment.end[axis]);
            }
            segment.trajectory = trajectory_values[row];
            segments_.push_back(segment);
        }
    }

    std::vector<std::vector<std::size_t>> spatial_groups(
        std::vector<std::size_t> ids,
        const std::vector<Vec3>& centers
    ) const {
        if (ids.size() <= static_cast<std::size_t>(branching_factor_)) {
            return {std::move(ids)};
        }
        const std::size_t capacity = static_cast<std::size_t>(branching_factor_);
        const std::size_t group_count = (ids.size() + capacity - 1) / capacity;
        const std::size_t x_slices = std::max<std::size_t>(1, static_cast<std::size_t>(
            std::ceil(std::cbrt(static_cast<double>(group_count)))
        ));
        std::stable_sort(ids.begin(), ids.end(), [&](std::size_t a, std::size_t b) {
            return centers[a][0] < centers[b][0];
        });
        const std::size_t x_size = (ids.size() + x_slices - 1) / x_slices;
        std::vector<std::vector<std::size_t>> groups;
        for (std::size_t x = 0; x < ids.size(); x += x_size) {
            const std::size_t x_end = std::min(x + x_size, ids.size());
            std::vector<std::size_t> slice(ids.begin() + x, ids.begin() + x_end);
            const std::size_t slice_groups = (slice.size() + capacity - 1) / capacity;
            const std::size_t y_slices = std::max<std::size_t>(1, static_cast<std::size_t>(
                std::ceil(std::sqrt(static_cast<double>(slice_groups)))
            ));
            std::stable_sort(slice.begin(), slice.end(), [&](std::size_t a, std::size_t b) {
                return centers[a][1] < centers[b][1];
            });
            const std::size_t y_size = (slice.size() + y_slices - 1) / y_slices;
            for (std::size_t y = 0; y < slice.size(); y += y_size) {
                const std::size_t y_end = std::min(y + y_size, slice.size());
                std::vector<std::size_t> slab(slice.begin() + y, slice.begin() + y_end);
                std::stable_sort(slab.begin(), slab.end(), [&](std::size_t a, std::size_t b) {
                    return centers[a][2] < centers[b][2];
                });
                for (std::size_t start = 0; start < slab.size(); start += capacity) {
                    const std::size_t end = std::min(start + capacity, slab.size());
                    groups.emplace_back(slab.begin() + start, slab.begin() + end);
                }
            }
        }
        return groups;
    }

    void kd_order(
        std::vector<std::size_t>& ids,
        std::size_t begin,
        std::size_t end,
        const std::vector<Vec3>& centers
    ) const {
        if (end - begin <= static_cast<std::size_t>(branching_factor_)) {
            return;
        }
        Vec3 low = centers[ids[begin]];
        Vec3 high = low;
        for (std::size_t i = begin + 1; i < end; ++i) {
            for (int axis = 0; axis < 3; ++axis) {
                low[axis] = std::min(low[axis], centers[ids[i]][axis]);
                high[axis] = std::max(high[axis], centers[ids[i]][axis]);
            }
        }
        int axis = 0;
        if (high[1] - low[1] > high[axis] - low[axis]) axis = 1;
        if (high[2] - low[2] > high[axis] - low[axis]) axis = 2;
        const std::size_t middle = begin + (end - begin) / 2;
        std::nth_element(
            ids.begin() + begin, ids.begin() + middle, ids.begin() + end,
            [&](std::size_t a, std::size_t b) { return centers[a][axis] < centers[b][axis]; }
        );
        kd_order(ids, begin, middle, centers);
        kd_order(ids, middle, end, centers);
    }

    std::size_t append_leaf(const std::vector<std::size_t>& item_ids) {
        Node node;
        node.entries.resize(static_cast<std::size_t>(branching_factor_));
        node.leaf = true;
        node.weight = item_ids.size();
        node.box = segments_[item_ids.front()].box;
        for (std::size_t i = 0; i < item_ids.size(); ++i) {
            const std::size_t item = item_ids[i];
            node.entries[i] = Entry{
                segments_[item].box, -1, static_cast<std::ptrdiff_t>(item), true
            };
            node.box = union_boxes(node.box, segments_[item].box);
        }
        nodes_.push_back(std::move(node));
        return nodes_.size() - 1;
    }

    std::size_t append_parent(const std::vector<std::size_t>& children) {
        Node node;
        node.entries.resize(static_cast<std::size_t>(branching_factor_));
        node.box = nodes_[children.front()].box;
        node.height = nodes_[children.front()].height + 1;
        for (std::size_t i = 0; i < children.size(); ++i) {
            const std::size_t child = children[i];
            node.entries[i] = Entry{
                nodes_[child].box, static_cast<std::ptrdiff_t>(child), -1, true
            };
            node.box = union_boxes(node.box, nodes_[child].box);
            node.weight += nodes_[child].weight;
        }
        nodes_.push_back(std::move(node));
        return nodes_.size() - 1;
    }

    std::vector<std::vector<std::size_t>> group_nodes(
        const std::vector<std::size_t>& level
    ) const {
        std::vector<Vec3> centers(nodes_.size());
        for (std::size_t id : level) centers[id] = center(nodes_[id].box);
        if (kind_ == TreeKind::KDTree) {
            std::vector<std::size_t> order = level;
            kd_order(order, 0, order.size(), centers);
            std::vector<std::vector<std::size_t>> groups;
            const std::size_t capacity = static_cast<std::size_t>(branching_factor_);
            for (std::size_t i = 0; i < order.size(); i += capacity) {
                groups.emplace_back(order.begin() + i, order.begin() + std::min(i + capacity, order.size()));
            }
            return groups;
        }
        return spatial_groups(level, centers);
    }

    void build() {
        std::vector<std::size_t> ids(segments_.size());
        std::iota(ids.begin(), ids.end(), 0);
        std::vector<std::vector<std::size_t>> leaf_groups;
        if (kind_ == TreeKind::TBTree) {
            std::stable_sort(ids.begin(), ids.end(), [&](std::size_t a, std::size_t b) {
                return segments_[a].trajectory < segments_[b].trajectory;
            });
            std::size_t start = 0;
            while (start < ids.size()) {
                std::size_t end = start + 1;
                while (end < ids.size() &&
                       segments_[ids[end]].trajectory == segments_[ids[start]].trajectory) {
                    ++end;
                }
                for (std::size_t i = start; i < end; i += branching_factor_) {
                    leaf_groups.emplace_back(
                        ids.begin() + i,
                        ids.begin() + std::min(i + static_cast<std::size_t>(branching_factor_), end)
                    );
                }
                start = end;
            }
        } else {
            std::vector<Vec3> centers(segments_.size());
            for (std::size_t i = 0; i < segments_.size(); ++i) centers[i] = center(segments_[i].box);
            if (kind_ == TreeKind::KDTree) {
                kd_order(ids, 0, ids.size(), centers);
                for (std::size_t i = 0; i < ids.size(); i += branching_factor_) {
                    leaf_groups.emplace_back(
                        ids.begin() + i,
                        ids.begin() + std::min(i + static_cast<std::size_t>(branching_factor_), ids.size())
                    );
                }
            } else {
                leaf_groups = spatial_groups(std::move(ids), centers);
            }
        }
        std::vector<std::size_t> level;
        level.reserve(leaf_groups.size());
        for (const auto& group : leaf_groups) level.push_back(append_leaf(group));
        while (level.size() > 1) {
            std::vector<std::size_t> next;
            for (const auto& group : group_nodes(level)) next.push_back(append_parent(group));
            level = std::move(next);
        }
        root_ = level.front();
    }

    std::uint64_t classical_count_with_containment(
        std::size_t node_id,
        const Box3& query,
        CountAuditStats& stats
    ) const {
        ++stats.node_reads;
        const Node& node = nodes_[node_id];
        if (contains(query, node.box)) {
            ++stats.contained_shortcuts;
            stats.records_via_shortcut += static_cast<std::uint64_t>(node.weight);
            return static_cast<std::uint64_t>(node.weight);
        }
        std::uint64_t count = 0;
        for (const Entry& entry : node.entries) {
            // Slot scans, valid-entry examinations, and exact geometry tests
            // are separate counters, not a combined physical-I/O estimate.
            ++stats.entry_slot_reads;
            if (!entry.valid) continue;
            ++stats.valid_entries_examined;
            if (!intersects(entry.box, query)) continue;
            if (entry.child >= 0) {
                count += classical_count_with_containment(
                    static_cast<std::size_t>(entry.child), query, stats
                );
            } else {
                ++stats.exact_segment_tests;
                if (segment_intersects(segments_[static_cast<std::size_t>(entry.item)], query)) {
                    ++count;
                }
            }
        }
        return count;
    }

    std::uint64_t classical_count(
        std::size_t node_id,
        const Box3& query,
        std::uint64_t& io_count
    ) const {
        ++io_count;
        const Node& node = nodes_[node_id];
        std::uint64_t count = 0;
        for (const Entry& entry : node.entries) {
            if (!entry.valid || !intersects(entry.box, query)) continue;
            if (entry.child >= 0) {
                count += classical_count(static_cast<std::size_t>(entry.child), query, io_count);
            } else if (segment_intersects(segments_[static_cast<std::size_t>(entry.item)], query)) {
                ++count;
            }
        }
        return count;
    }

    void collect_ids(
        std::size_t node_id,
        const Box3& query,
        std::vector<std::size_t>& result
    ) const {
        const Node& node = nodes_[node_id];
        for (const Entry& entry : node.entries) {
            if (!entry.valid || !intersects(entry.box, query)) continue;
            if (entry.child >= 0) {
                collect_ids(static_cast<std::size_t>(entry.child), query, result);
            } else if (segment_intersects(segments_[static_cast<std::size_t>(entry.item)], query)) {
                result.push_back(static_cast<std::size_t>(entry.item));
            }
        }
    }

    void collect_subtree_ids(
        std::size_t node_id,
        std::vector<std::size_t>& result
    ) const {
        for (const Entry& entry : nodes_[node_id].entries) {
            if (!entry.valid) continue;
            if (entry.child >= 0) {
                collect_subtree_ids(static_cast<std::size_t>(entry.child), result);
            } else {
                result.push_back(static_cast<std::size_t>(entry.item));
            }
        }
    }

    // This discovery never obtains k or sums candidate weights. A successful
    // witness retains the entire current frontier, not only its scanned prefix.
    void sampling_discover(
        const Box3& query,
        const std::vector<std::size_t>& candidates,
        int depth,
        SamplingDiscovery& result
    ) const {
        if (candidates.empty()) return;
        std::vector<std::size_t> next;
        for (std::size_t node_id : candidates) {
            ++result.node_reads;
            const Node& node = nodes_[node_id];
            for (const Entry& entry : node.entries) {
                ++result.scan_slots;
                if (!entry.valid) continue;
                bool witness;
                if (entry.child >= 0) {
                    witness = contains(query, entry.box);
                    if (!witness && intersects(entry.box, query)) {
                        next.push_back(static_cast<std::size_t>(entry.child));
                    }
                } else {
                    ++result.geometry_tests;
                    witness = segment_intersects(segments_[static_cast<std::size_t>(entry.item)], query);
                }
                if (witness) {
                    result.frontier = candidates;
                    result.height = node.height;
                    result.depth = depth;
                    result.kind = entry.child >= 0 ? "contained_subtree" : "leaf_segment";
                    return;
                }
            }
        }
        sampling_discover(query, next, depth + 1, result);
    }

    static std::size_t sampling_dummy() {
        return std::numeric_limits<std::size_t>::max();
    }

    template <typename ChooseSlot>
    std::size_t sampling_route_padded(
        std::size_t node_id,
        int height,
        ChooseSlot&& choose_slot,
        SamplingLocalStats& stats
    ) const {
        for (int level = 0; level < height; ++level) {
            ++stats.node_reads;
            const Entry& entry = nodes_[node_id].entries[choose_slot()];
            if (!entry.valid) return sampling_dummy();
            if (entry.child >= 0) {
                node_id = static_cast<std::size_t>(entry.child);
            } else {
                if (level + 1 != height) {
                    throw std::logic_error("padded proposal encountered an unexpected leaf height");
                }
                return static_cast<std::size_t>(entry.item);
            }
        }
        throw std::logic_error("padded proposal did not terminate at a record");
    }

    std::size_t sampling_route_rank(
        const std::vector<std::size_t>& frontier,
        const std::vector<std::size_t>& cumulative_weights,
        std::size_t rank,
        SamplingLocalStats& stats
    ) const {
        const auto found = std::upper_bound(cumulative_weights.begin(), cumulative_weights.end(), rank);
        if (found == cumulative_weights.end()) {
            throw std::logic_error("proposal rank exceeds prepared frontier weight");
        }
        const std::size_t root_index = static_cast<std::size_t>(found - cumulative_weights.begin());
        if (root_index > 0) rank -= cumulative_weights[root_index - 1];
        std::size_t node_id = frontier[root_index];
        while (true) {
            ++stats.node_reads;
            const Node& node = nodes_[node_id];
            bool descended = false;
            for (const Entry& entry : node.entries) {
                if (!entry.valid) continue;
                if (entry.child >= 0) {
                    ++stats.weight_reads;
                    const std::size_t child = static_cast<std::size_t>(entry.child);
                    const std::size_t weight = nodes_[child].weight;
                    if (rank < weight) {
                        node_id = child;
                        descended = true;
                        break;
                    }
                    rank -= weight;
                } else {
                    if (rank == 0) return static_cast<std::size_t>(entry.item);
                    --rank;
                }
            }
            if (!descended) {
                throw std::logic_error("proposal rank disagrees with actual subtree records");
            }
        }
    }

    py::dict sampling_audit_support(
        const Box3& query,
        const SamplingDiscovery& discovery,
        const std::vector<std::size_t>& cumulative_weights,
        std::size_t records,
        bool valid_profile,
        bool enabled,
        std::uint64_t limit,
        double proposal_domain
    ) const {
        py::dict result(
            "support_verified"_a = false,
            "support_status"_a = enabled ? "not_verified_domain_limit" : "disabled",
            "support_domain"_a = proposal_domain,
            "support_dummy_count"_a = py::none(),
            "support_record_ids"_a = py::none(),
            "support_multiplicities"_a = py::none(),
            "support_hit_ids"_a = py::none(),
            "support_routing_node_reads"_a = py::none(),
            "support_routing_weight_reads"_a = py::none(),
            "support_geometry_tests"_a = py::none()
        );
        if (!enabled) return result;
        std::uint64_t domain = valid_profile ? static_cast<std::uint64_t>(records)
            : static_cast<std::uint64_t>(discovery.frontier.size());
        if (domain > limit) return result;
        if (!valid_profile) {
            const std::uint64_t block = static_cast<std::uint64_t>(branching_factor_);
            for (int level = 0; level < discovery.height; ++level) {
                if (domain > limit / block) return result;
                domain *= block;
            }
        }
        // Enumerate actual ranks or complete (frontier root, B-ary path)
        // choices through the SAME routing functions used above. A dummy hit
        // represents every complete path sharing that early dummy prefix.
        SamplingLocalStats diagnostic;
        std::vector<std::size_t> proposals;
        std::uint64_t dummy_count = 0;
        const std::uint64_t per_root = discovery.frontier.empty() ? 0
            : domain / static_cast<std::uint64_t>(discovery.frontier.size());
        for (std::uint64_t choice = 0; choice < domain; ++choice) {
            std::size_t item;
            if (valid_profile) {
                item = sampling_route_rank(discovery.frontier, cumulative_weights,
                    static_cast<std::size_t>(choice), diagnostic);
            } else {
                std::uint64_t digits = choice % per_root;
                item = sampling_route_padded(
                    discovery.frontier[static_cast<std::size_t>(choice / per_root)], discovery.height,
                    [&]() {
                        const std::size_t slot = static_cast<std::size_t>(digits % branching_factor_);
                        digits /= static_cast<std::uint64_t>(branching_factor_);
                        return slot;
                    }, diagnostic
                );
            }
            if (item == sampling_dummy()) ++dummy_count;
            else proposals.push_back(item);
        }
        std::sort(proposals.begin(), proposals.end());
        std::vector<std::size_t> ids, hit_ids;
        std::vector<std::uint64_t> multiplicities;
        for (std::size_t item : proposals) {
            if (ids.empty() || ids.back() != item) {
                ids.push_back(item);
                multiplicities.push_back(1);
                if (segment_intersects(segments_[item], query)) hit_ids.push_back(item);
            } else {
                ++multiplicities.back();
            }
        }
        result["support_verified"] = true;
        result["support_status"] = "verified";
        result["support_domain"] = domain;
        result["support_dummy_count"] = dummy_count;
        result["support_record_ids"] = ids;
        result["support_multiplicities"] = multiplicities;
        result["support_hit_ids"] = hit_ids;
        result["support_routing_node_reads"] = diagnostic.node_reads;
        result["support_routing_weight_reads"] = diagnostic.weight_reads;
        result["support_geometry_tests"] = ids.size();
        return result;
    }

    void audit_frontier(
        const Box3& query,
        const std::vector<std::size_t>& candidates,
        int depth,
        bool verify_coverage,
        AuditFrontier& audit
    ) const {
        if (candidates.empty()) return;
        const int height = nodes_[candidates.front()].height;
        std::size_t weight = 0;
        for (std::size_t node_id : candidates) weight += nodes_[node_id].weight;
        std::vector<std::size_t> next;
        // Deliberately mirror quantum_query's candidate and physical-slot
        // order, pruning rule, and first-witness return without changing it.
        for (std::size_t node_id : candidates) {
            const Node& node = nodes_[node_id];
            for (const Entry& entry : node.entries) {
                ++audit.scan_slots;
                if (!entry.valid) continue;
                bool witness = false;
                if (entry.child >= 0) {
                    witness = contains(query, entry.box);
                    if (!witness && intersects(entry.box, query)) {
                        next.push_back(static_cast<std::size_t>(entry.child));
                    }
                } else {
                    witness = segment_intersects(
                        segments_[static_cast<std::size_t>(entry.item)], query
                    );
                }
                if (witness) {
                    audit.frontier_nodes = candidates.size();
                    audit.frontier_records = weight;
                    audit.witness_height = height;
                    audit.witness_depth = depth;
                    audit.witness_kind = entry.child >= 0
                        ? "contained_subtree" : "leaf_segment";
                    if (verify_coverage) audit.witness_frontier = candidates;
                    return;
                }
            }
        }
        audit_frontier(query, next, depth + 1, verify_coverage, audit);
    }

    static std::uint64_t author_post_selection_cost(
        int height,
        std::size_t weight,
        std::uint64_t answer_count,
        std::mt19937_64& random
    ) {
        if (height <= 0 || weight == 0 || answer_count == 0) return 0;
        std::uniform_int_distribution<std::size_t> distribution(0, weight - 1);
        std::uint64_t cost = static_cast<std::uint64_t>(height);
        while (distribution(random) > answer_count) {
            cost += static_cast<std::uint64_t>(height);
        }
        return cost;
    }

    void quantum_query(
        const Box3& query,
        const std::vector<std::size_t>& candidates,
        std::uint64_t answer_count,
        std::uint64_t& io_count,
        double& probability_aware_io,
        std::mt19937_64& random
    ) const {
        if (candidates.empty()) return;
        const int height = nodes_[candidates.front()].height;
        std::size_t weight = 0;
        for (std::size_t node_id : candidates) weight += nodes_[node_id].weight;
        std::vector<std::size_t> next;
        for (std::size_t node_id : candidates) {
            const Node& node = nodes_[node_id];
            for (const Entry& entry : node.entries) {
                ++io_count;  // author-compatible: every physical slot is scanned
                probability_aware_io += 1.0;
                if (!entry.valid) continue;
                bool witness = false;
                if (entry.child >= 0) {
                    witness = contains(query, entry.box);
                    if (!witness && intersects(entry.box, query)) {
                        next.push_back(static_cast<std::size_t>(entry.child));
                    }
                } else {
                    witness = segment_intersects(
                        segments_[static_cast<std::size_t>(entry.item)], query
                    );
                }
                if (witness) {
                    io_count += author_post_selection_cost(
                        height, weight, answer_count, random
                    );
                    const double success = std::min(
                        1.0,
                        static_cast<double>(answer_count) / static_cast<double>(weight)
                    );
                    if (success > 0.0) {
                        probability_aware_io += static_cast<double>(height) / success;
                    }
                    return;
                }
            }
        }
        quantum_query(
            query, next, answer_count, io_count, probability_aware_io, random
        );
    }

    TrajectoryQueryResult3D query_with_rng(
        const Box3& query,
        std::mt19937_64& random
    ) const {
        std::uint64_t classical_io = 0;
        const std::uint64_t count = classical_count(root_, query, classical_io);
        std::uint64_t quantum_io = 0;
        double probability_io = 0.0;
        quantum_query(
            query, std::vector<std::size_t>{root_}, count,
            quantum_io, probability_io, random
        );
        return TrajectoryQueryResult3D{
            count, classical_io, quantum_io, probability_io
        };
    }
};

void bind_trajectory_trees_3d(py::module_& module) {
    py::class_<TrajectoryQueryResult3D>(module, "TrajectoryQueryResult3D", py::module_local())
        .def_readonly("result_count", &TrajectoryQueryResult3D::result_count)
        .def_readonly("classical_io", &TrajectoryQueryResult3D::classical_io)
        .def_readonly("quantum_io", &TrajectoryQueryResult3D::quantum_io)
        .def_readonly("probability_aware_io", &TrajectoryQueryResult3D::probability_aware_io)
        .def_property_readonly("model_speedup", &TrajectoryQueryResult3D::model_speedup)
        .def("as_dict", [](const TrajectoryQueryResult3D& result) {
            return py::dict(
                "result_count"_a = result.result_count,
                "classical_io"_a = result.classical_io,
                "quantum_io"_a = result.quantum_io,
                "probability_aware_io"_a = result.probability_aware_io,
                "model_speedup"_a = result.model_speedup()
            );
        });

    py::class_<TrajectoryBenchmarkResult3D>(module, "TrajectoryBenchmarkResult3D", py::module_local())
        .def_readonly("query_count", &TrajectoryBenchmarkResult3D::query_count)
        .def_readonly("average_answers", &TrajectoryBenchmarkResult3D::average_answers)
        .def_readonly("average_classical_io", &TrajectoryBenchmarkResult3D::average_classical_io)
        .def_readonly("average_quantum_io", &TrajectoryBenchmarkResult3D::average_quantum_io)
        .def_readonly("average_probability_aware_io", &TrajectoryBenchmarkResult3D::average_probability_aware_io)
        .def_property_readonly("model_speedup", &TrajectoryBenchmarkResult3D::model_speedup);

    py::class_<QuantumTrajectoryTree3D>(module, "QuantumTrajectoryTree3D", py::module_local())
        .def(
            py::init<
                const py::array_t<double, py::array::c_style | py::array::forcecast>&,
                const py::array_t<double, py::array::c_style | py::array::forcecast>&,
                const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>&,
                int,
                const std::string&
            >(),
            py::arg("starts"), py::arg("ends"), py::arg("trajectory_ids"),
            py::arg("branching_factor") = 16,
            py::arg("tree_type") = "packed_rtree"
        )
        .def(
            "query", &QuantumTrajectoryTree3D::query,
            py::arg("xmin"), py::arg("xmax"),
            py::arg("ymin"), py::arg("ymax"),
            py::arg("tmin"), py::arg("tmax"),
            py::arg("seed") = 0
        )
        .def(
            "query_ids", &QuantumTrajectoryTree3D::query_ids,
            py::arg("xmin"), py::arg("xmax"),
            py::arg("ymin"), py::arg("ymax"),
            py::arg("tmin"), py::arg("tmax")
        )
        .def(
            "count_audit_query", &QuantumTrajectoryTree3D::count_audit_query,
            py::arg("xmin"), py::arg("xmax"),
            py::arg("ymin"), py::arg("ymax"),
            py::arg("tmin"), py::arg("tmax"),
            "Count exact segment hits using contained-node cardinality shortcuts; "
            "report node reads, physical-slot scans, and geometry tests separately."
        )
        .def(
            "sample_query", &QuantumTrajectoryTree3D::sample_query,
            py::arg("xmin"), py::arg("xmax"),
            py::arg("ymin"), py::arg("ymax"),
            py::arg("tmin"), py::arg("tmax"),
            py::arg("proposal_profile") = "padded",
            py::arg("seed") = 0,
            py::arg("samples") = 8,
            py::arg("max_trials") = 1000000,
            py::arg("audit_support") = false,
            py::arg("support_limit") = 4096,
            "Draw real record proposals and reject by exact geometry without known k. "
            "Discovery is shared; max_trials caps the complete call. Actual node/weight "
            "reads and fixed-height model costs are distinct; support enumeration is opt-in."
        )
        .def(
            "audit_query", &QuantumTrajectoryTree3D::audit_query,
            py::arg("xmin"), py::arg("xmax"),
            py::arg("ymin"), py::arg("ymax"),
            py::arg("tmin"), py::arg("tmax"),
            py::arg("verify_coverage") = false,
            "Audit deterministic costs at the first witness frontier. Coverage "
            "is None unless explicitly verified; no-witness depth is -1 and "
            "all no-witness frontier/domain fields are zero."
        )
        .def(
            "benchmark", &QuantumTrajectoryTree3D::benchmark,
            py::arg("queries"), py::arg("seed") = 0
        )
        .def("matched_layout", &QuantumTrajectoryTree3D::matched_layout)
        .def("matched_query", &QuantumTrajectoryTree3D::matched_query)
        .def_property_readonly("size", &QuantumTrajectoryTree3D::size)
        .def_property_readonly("node_count", &QuantumTrajectoryTree3D::node_count)
        .def_property_readonly("height", &QuantumTrajectoryTree3D::height)
        .def_property_readonly("branching_factor", &QuantumTrajectoryTree3D::branching_factor)
        .def_property_readonly("tree_type", &QuantumTrajectoryTree3D::tree_type);
}

PYBIND11_MODULE(_qrq_matched, module) { bind_trajectory_trees_3d(module); }
