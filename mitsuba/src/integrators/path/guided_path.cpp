/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob
    Copyright (c) 2017 by ETH Zurich, Thomas Mueller.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/statistics.h>

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <iomanip>
#include <sstream>

MTS_NAMESPACE_BEGIN

class BlobWriter {
public:
    BlobWriter(const std::string& filename)
        : f(filename, std::ios::out | std::ios::binary) {
    }

    template <typename Type>
    typename std::enable_if<std::is_standard_layout<Type>::value, BlobWriter&>::type
        operator << (Type Element) {
        Write(&Element, 1);
        return *this;
    }

    // CAUTION: This function may break down on big-endian architectures.
    //          The ordering of bytes has to be reverted then.
    template <typename T>
    void Write(T* Src, size_t Size) {
        f.write(reinterpret_cast<const char*>(Src), Size * sizeof(T));
    }

private:
    std::ofstream f;
};

static void addToAtomicFloat(std::atomic<Float>& var, Float val) {
    auto current = var.load();
    while (!var.compare_exchange_weak(current, current + val));
}

class QuadTreeNode {
public:
    QuadTreeNode() {
        m_children = {};
        for (size_t i = 0; i < m_mean.size(); ++i) {
            m_mean[i].store(0, std::memory_order_relaxed);
        }
    }

    void setMean(int index, Float val) {
        m_mean[index].store(val, std::memory_order_relaxed);
    }

    Float mean(int index) const {
        return m_mean[index].load(std::memory_order_relaxed);
    }

    void copyFrom(const QuadTreeNode& arg) {
        for (int i = 0; i < 4; ++i) {
            setMean(i, arg.mean(i));
            m_children[i] = arg.m_children[i];
        }
    }

    QuadTreeNode(const QuadTreeNode& arg) {
        copyFrom(arg);
    }

    QuadTreeNode& operator=(const QuadTreeNode& arg) {
        copyFrom(arg);
        return *this;
    }

    void setChild(int idx, uint16_t val) {
        m_children[idx] = val;
    }

    uint16_t child(int idx) const {
        return m_children[idx];
    }

    void setMean(Float meanVal) {
        for (int i = 0; i < 4; ++i) {
            setMean(i, meanVal);
        }
    }

    int childIndex(Point2& p) const {
        int res = 0;
        for (int i = 0; i < Point2::dim; ++i) {
            if (p[i] < 0.5f) {
                p[i] *= 2;
            } else {
                p[i] = (p[i] - 0.5f) * 2;
                res |= 1 << i;
            }
        }

        return res;
    }

    Float eval(Point2& p, const std::vector<QuadTreeNode>& nodes) const {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (isLeaf(index)) {
            return 4 * mean(index);
        } else {
            return 4 * nodes[child(index)].eval(p, nodes);
        }
    }

    Float pdf(Point2& p, const std::vector<QuadTreeNode>& nodes) const {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (!(mean(index) > 0)) {
            return 0;
        }

        const Float factor = 4 * mean(index) / (mean(0) + mean(1) + mean(2) + mean(3));
        if (isLeaf(index)) {
            return factor;
        } else {
            return factor * nodes[child(index)].pdf(p, nodes);
        }
    }

    int depthAt(Point2& p, const std::vector<QuadTreeNode>& nodes) const {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (isLeaf(index)) {
            return 1;
        } else {
            return 1 + nodes[child(index)].depthAt(p, nodes);
        }
    }

    Point2 sample(Point2 sample, const std::vector<QuadTreeNode>& nodes) const {
        int index = 0;

        Float topLeft = mean(0);
        Float topRight = mean(1);
        Float partial = topLeft + mean(2);
        Float total = partial + topRight + mean(3);

        SAssert(total > 0.0f);

        Float boundary = partial / total;
        Point2 origin = Point2{0.0f, 0.0f};

        if (sample.x < boundary) {
            SAssert(partial > 0);
            sample.x /= boundary;
            boundary = topLeft / partial;
        } else {
            partial = total - partial;
            SAssert(partial > 0);
            origin.x = 0.5f;
            sample.x = (sample.x - boundary) / (1.0f - boundary);
            boundary = topRight / partial;
            index |= 1 << 0;
        }

        if (sample.y < boundary) {
            sample.y /= boundary;
        } else {
            origin.y = 0.5f;
            sample.y = (sample.y - boundary) / (1.0f - boundary);
            index |= 1 << 1;
        }

        if (isLeaf(index)) {
            return origin + 0.5f * sample;
        } else {
            return origin + 0.5f * nodes[child(index)].sample(sample, nodes);
        }
    }

    void record(Point2& p, Float radiance, std::vector<QuadTreeNode>& nodes) {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        int index = childIndex(p);

        addToAtomicFloat(m_mean[index], radiance);

        if (!isLeaf(index)) {
            nodes[child(index)].record(p, radiance, nodes);
        }
    }

    bool isLeaf(int index) const {
        return child(index) == 0;
    }

private:
    std::array<std::atomic<Float>, 4> m_mean;
    std::array<uint16_t, 4> m_children;
};


class DTree {
public:
    DTree() {
        m_atomic.sum.store(1, std::memory_order_relaxed);
        m_maxDepth = 0;
        m_nodes.emplace_back();
        m_nodes.front().setMean(1.0f / 4);
    }

    const QuadTreeNode& node(size_t i) const {
        return m_nodes[i];
    }

    Float mean() const {
        if (m_atomic.nSamples == 0) {
            return 0;
        }
        const Float factor = 1 / (M_PI * 4 * m_atomic.nSamples);
        return factor * m_atomic.sum;
    }

    Spectrum getMeasurement() const {
        return m_atomic.getMeasurement();
    }

    void recordRadiance(Point pos, Point2 p, Float radiance, bool count) {

        if (!std::isfinite(radiance)) {
            radiance = 0;
        }

        if (count) {
            ++m_atomic.nSamples;
        }

        addToAtomicFloat(m_atomic.sum, radiance);

        m_nodes[0].record(p, radiance, m_nodes);
    }

    void recordMeasurement(const Spectrum& m) {
        m_atomic.recordMeasurement(m);
    }

    Float evalRadiance(Point2 p) const {
        if (m_atomic.nSamples == 0) {
            return 0;
        }

        return m_nodes[0].eval(p, m_nodes) / (M_PI * m_atomic.nSamples);
    }

    Float evalPdf(Point2 p) const {
        if (!(mean() > 0)) {
            return 1 / (4 * M_PI);
        }

        return m_nodes[0].pdf(p, m_nodes) / (4 * M_PI);
    }

    int depthAt(Point2 p) const {
        return m_nodes[0].depthAt(p, m_nodes);
    }

    int depth() const {
        return m_maxDepth;
    }

    Point2 sampleRadiance(Point2 sample) const {
        if (!(mean() > 0)) {
            return sample;
        }

        Point2 res = m_nodes[0].sample(sample, m_nodes);
        SAssert(res.x >= 0 && res.x <= 1 && res.y >= 0 && res.y <= 1);

        return res;
    }

    size_t amountNodes() const {
        return m_nodes.size();
    }

    size_t amountSamples() const {
        return m_atomic.nSamples;
    }

    void setSamples(size_t nSamples) {
        m_atomic.nSamples = nSamples;
    }

    void reset(const DTree& previousDTree, int newMaxDepth, Float subdivisionThreshold) {
        m_atomic = Atomic{};
        m_maxDepth = 0;
        m_nodes.clear();
        m_nodes.emplace_back();

        struct StackNode {
            size_t nodeIndex;
            size_t otherNodeIndex;
            const DTree* otherDTree;
            int depth;
        };

        std::stack<StackNode> nodeIndices;
        nodeIndices.push({0, 0, &previousDTree, 1});

        // Create the topology of the new DTree to be the refined version
        // of the previous DTree. Subdivision is recursive if enough energy is there.
        while (!nodeIndices.empty()) {
            StackNode sNode = nodeIndices.top();
            nodeIndices.pop();

            m_maxDepth = std::max(m_maxDepth, sNode.depth);

            for (int i = 0; i < 4; ++i) {
                const QuadTreeNode& otherNode = sNode.otherDTree->m_nodes[sNode.otherNodeIndex];
                const Float total = previousDTree.m_atomic.sum;
                const Float fraction = total > 0 ? (otherNode.mean(i) / total) : 0;
                SAssert(fraction <= 1.0f + Epsilon);

                if (sNode.depth < newMaxDepth && fraction > subdivisionThreshold) {
                    if (!otherNode.isLeaf(i)) {
                        SAssert(sNode.otherDTree == &previousDTree);
                        nodeIndices.push({m_nodes.size(), otherNode.child(i), &previousDTree, sNode.depth + 1});
                    } else {
                        nodeIndices.push({m_nodes.size(), m_nodes.size(), this, sNode.depth + 1});
                    }

                    m_nodes[sNode.nodeIndex].setChild(i, static_cast<uint16_t>(m_nodes.size()));
                    m_nodes.emplace_back();
                    m_nodes.back().setMean(otherNode.mean(i) / 4);

                    if (m_nodes.size() > std::numeric_limits<uint16_t>::max()) {
                        SLog(EWarn, "DTreeWrapper hit maximum children amount.");
                        nodeIndices = std::stack<StackNode>();
                        break;
                    }
                }
            }
        }

        // Uncomment once memory becomes an issue.
        //m_nodes.shrink_to_fit();

        for (auto& node : m_nodes) {
            node.setMean(0);
        }
    }

    size_t approxMemoryFootprint() const {
        return m_nodes.capacity() * sizeof(QuadTreeNode) + sizeof(*this);
    }

private:
    std::vector<QuadTreeNode> m_nodes;

    struct Atomic {
        Atomic() {
            sum.store(0, std::memory_order_relaxed);
            nSamples = 0;

            measurementR.store(0, std::memory_order_relaxed);
            measurementG.store(0, std::memory_order_relaxed);
            measurementB.store(0, std::memory_order_relaxed);
            nMeasurementSamples = 0;
        }

        void copyFrom(const Atomic& arg) {
            sum.store(arg.sum.load(std::memory_order_relaxed), std::memory_order_relaxed);
            nSamples = arg.nSamples.load(std::memory_order_relaxed);

            measurementR.store(arg.measurementR.load(std::memory_order_relaxed), std::memory_order_relaxed);
            measurementG.store(arg.measurementG.load(std::memory_order_relaxed), std::memory_order_relaxed);
            measurementB.store(arg.measurementB.load(std::memory_order_relaxed), std::memory_order_relaxed);
            nMeasurementSamples = arg.nMeasurementSamples.load(std::memory_order_relaxed);
        }

        Atomic(const Atomic& arg) {
            copyFrom(arg);
        }

        Atomic& operator=(const Atomic& arg) {
            copyFrom(arg);
            return *this;
        }

        std::atomic<Float> sum;
        std::atomic<size_t> nSamples;

        void recordMeasurement(const Spectrum& val) {
            ++nMeasurementSamples;
            addToAtomicFloat(measurementR, val[0]);
            addToAtomicFloat(measurementG, val[1]);
            addToAtomicFloat(measurementB, val[2]);
        }

        Spectrum getMeasurement() const {

            size_t n = nMeasurementSamples.load(std::memory_order_relaxed);
            if (n == 0) {
                return Spectrum{0.0f};
            }

            Spectrum result;
            result[0] = measurementR.load(std::memory_order_relaxed);
            result[1] = measurementG.load(std::memory_order_relaxed);
            result[2] = measurementB.load(std::memory_order_relaxed);
            return result / (Float)n;
        }

        std::atomic<Float> measurementR;
        std::atomic<Float> measurementG;
        std::atomic<Float> measurementB;
        std::atomic<size_t> nMeasurementSamples;

    } m_atomic;

    int m_maxDepth;
};


struct DTreeWrapper {
public:
    DTreeWrapper() {
    }

    void recordRadiance(const Point& pos, const Vector& dir, Spectrum radiance, Point2 sample, bool shouldCount = true) {
        building.recordRadiance(pos, dirToCanonical(dir), radiance.average(), shouldCount);
    }

    void recordMeasurement(Spectrum m) {
        if (!m.isValid()) {
            m = Spectrum{0.0f};
        }

        building.recordMeasurement(m);
    }

    static Vector canonicalToDir(Point2 p) {
        const Float cosTheta = 2 * p.x - 1;
        const Float phi = 2 * M_PI * p.y;

        const Float sinTheta = sqrt(1 - cosTheta * cosTheta);
        Float sinPhi, cosPhi;
        math::sincos(phi, &sinPhi, &cosPhi);

        return {sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
    }

    static Point2 dirToCanonical(const Vector& d) {
        if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z)) {
            return {0, 0};
        }

        const Float cosTheta = std::min(std::max(d.z, -1.0f), 1.0f);
        Float phi = std::atan2(d.y, d.x);
        while (phi < 0)
            phi += 2.0 * M_PI;

        return {(cosTheta + 1) / 2, phi / (2 * M_PI)};
    }

    Float estimateRadiance(Point2 p) const {
        return sampling.evalRadiance(p);
    }

    Float estimateRadiance(const Vector& d) const {
        return estimateRadiance(dirToCanonical(d));
    }

    void build() {
        sampling = building;
    }

    void reset(int maxDepth, Float subdivisionThreshold) {
        building.reset(sampling, maxDepth, subdivisionThreshold);
    }

    Vector sampleDirection(Point2 sample) const {
        return canonicalToDir(sampling.sampleRadiance(sample));
    }

    Float samplePdf(const Vector& dir) const {
        return sampling.evalPdf(dirToCanonical(dir));
    }

    Float diff(const DTreeWrapper& other) const {
        return 0.0f;
    }

    int depth() const {
        return sampling.depth();
    }

    size_t amountNodes() const {
        return sampling.amountNodes();
    }

    float meanRadiance() const {
        return sampling.mean();
    }

    Spectrum measurementEstimate() const {
        return sampling.getMeasurement();
    }

    size_t amountSamples() const {
        return sampling.amountSamples();
    }

    size_t amountSamplesBuilding() const {
        return building.amountSamples();
    }

    void setSamplesBuilding(size_t nSamples) {
        building.setSamples(nSamples);
    }

    size_t approxMemoryFootprint() const {
        return building.approxMemoryFootprint() + sampling.approxMemoryFootprint();
    }

    void dump(BlobWriter& blob, const Point& p, const Vector& size) const {
        blob
            << (float)p.x << (float)p.y << (float)p.z
            << (float)size.x << (float)size.y << (float)size.z
            << (float)sampling.mean() << (uint64_t)sampling.amountSamples() << (uint64_t)sampling.amountNodes();

        for (size_t i = 0; i < sampling.amountNodes(); ++i) {
            const auto& node = sampling.node(i);
            for (int j = 0; j < 4; ++j) {
                blob << (float)node.mean(j) << (uint16_t)node.child(j);
            }
        }
    }

private:
    DTree building;
    DTree sampling;
};


struct STreeNode {
    STreeNode() {
        children = {};
        isLeaf = true;
        axis = 0;
    }

    int childIndex(Point& p) const {
        if (p[axis] < 0.5f) {
            p[axis] *= 2;
            return 0;
        } else {
            p[axis] = (p[axis] - 0.5f) * 2;
            return 1;
        }
    }

    int nodeIndex(Point& p) const {
        return children[childIndex(p)];
    }

    DTreeWrapper* dTreeWrapper(Point& p, std::vector<STreeNode>& nodes) {
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf) {
            return &dTree;
        } else {
            return nodes[nodeIndex(p)].dTreeWrapper(p, nodes);
        }
    }

    const DTreeWrapper* dTreeWrapper() const {
        return &dTree;
    }

    int depth(Point& p, const std::vector<STreeNode>& nodes) const {
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf) {
            return 1;
        } else {
            return 1 + nodes[nodeIndex(p)].depth(p, nodes);
        }
    }

    int depth(const std::vector<STreeNode>& nodes) const {
        int result = 1;

        if (!isLeaf) {
            for (auto c : children) {
                result = std::max(result, 1 + nodes[c].depth(nodes));
            }
        }

        return result;
    }

    void forEachLeaf(
        std::function<void(const DTreeWrapper*, const Point&, const Vector&)> func,
        Point p, Vector size, const std::vector<STreeNode>& nodes) const {

        if (isLeaf) {
            func(&dTree, p, size);
        } else {
            size[axis] /= 2;
            for (int i = 0; i < 2; ++i) {
                Point childP = p;
                if (i == 1) {
                    childP[axis] += size[axis];
                }

                nodes[children[i]].forEachLeaf(func, childP, size, nodes);
            }
        }
    }

    bool isLeaf;
    DTreeWrapper dTree;
    int axis;
    std::array<uint32_t, 2> children;
};


class STree {
public:
    STree(const AABB& aabb) {
        clear();

        m_aabb = aabb;

        // Enlarge AABB to turn it into a cube. This has the effect
        // of nicer hierarchical subdivisions.
        Vector size = m_aabb.max - m_aabb.min;
        Float maxSize = std::max(std::max(size.x, size.y), size.z);
        m_aabb.max = m_aabb.min + Vector(maxSize);
    }

    void clear() {
        m_nodes.clear();
        m_nodes.emplace_back();
    }

    void subdivideAll() {
        int nNodes = (int)m_nodes.size();
        for (int i = 0; i < nNodes; ++i) {
            if (m_nodes[i].isLeaf) {
                subdivide(i, m_nodes);
            }
        }
    }

    void subdivide(int nodeIdx, std::vector<STreeNode>& nodes) {
        // Add 2 child nodes
        nodes.resize(nodes.size() + 2);

        if (nodes.size() > std::numeric_limits<uint32_t>::max()) {
            SLog(EWarn, "DTreeWrapper hit maximum children amount.");
            return;
        }

        STreeNode& cur = nodes[nodeIdx];
        for (int i = 0; i < 2; ++i) {
            uint32_t idx = (uint32_t)nodes.size() - 2 + i;
            cur.children[i] = idx;
            nodes[idx].axis = (cur.axis + 1) % 3;
            nodes[idx].dTree = cur.dTree;
            nodes[idx].dTree.setSamplesBuilding(nodes[idx].dTree.amountSamplesBuilding() / 2);
        }
        cur.isLeaf = false;
        cur.dTree = {}; // Reset to an empty dtree to save memory.
    }

    DTreeWrapper* dTreeWrapper(Point p) {
        Vector size = m_aabb.max - m_aabb.min;
        p = Point(p - m_aabb.min);
        p.x /= size.x;
        p.y /= size.y;
        p.z /= size.z;

        return m_nodes[0].dTreeWrapper(p, m_nodes);
    }

    void forEachDTreeWrapperConst(std::function<void(const DTreeWrapper*)> func) const {
        for (auto& node : m_nodes) {
            if (node.isLeaf) {
                func(&node.dTree);
            }
        }
    }

    void forEachDTreeWrapperConstP(std::function<void(const DTreeWrapper*, const Point&, const Vector&)> func) const {
        m_nodes[0].forEachLeaf(func, m_aabb.min, m_aabb.max - m_aabb.min, m_nodes);
    }

    void forEachDTreeWrapperParallel(std::function<void(DTreeWrapper*)> func) {
        int nDTreeWrappers = static_cast<int>(m_nodes.size());

#pragma omp parallel for
        for (int i = 0; i < nDTreeWrappers; ++i) {
            if (m_nodes[i].isLeaf) {
                func(&m_nodes[i].dTree);
            }
        }
    }

    void dump(BlobWriter& blob) const {
        forEachDTreeWrapperConstP([&blob](const DTreeWrapper* dTree, const Point& p, const Vector& size) {
            if (dTree->amountSamples() > 0) {
                dTree->dump(blob, p, size);
            }
        });
    }

    bool shallSplit(const STreeNode& node, int depth, size_t samplesRequired) {
        return m_nodes.size() < std::numeric_limits<uint32_t>::max() - 1 && node.dTree.amountSamplesBuilding() > samplesRequired;
    }

    void refine(size_t sTreeThreshold, int maxMB) {
        if (maxMB >= 0) {
            size_t approxMemoryFootprint = 0;
            for (const auto& node : m_nodes) {
                approxMemoryFootprint += node.dTreeWrapper()->approxMemoryFootprint();
            }

            if (approxMemoryFootprint / 1000000 >= (size_t)maxMB) {
                return;
            }
        }
        
        struct StackNode {
            size_t index;
            int depth;
        };

        std::stack<StackNode> nodeIndices;
        nodeIndices.push({0,  1});
        while (!nodeIndices.empty()) {
            StackNode sNode = nodeIndices.top();
            nodeIndices.pop();

            // Subdivide if needed and leaf
            if (m_nodes[sNode.index].isLeaf) {
                if (shallSplit(m_nodes[sNode.index], sNode.depth, sTreeThreshold)) {
                    subdivide((int)sNode.index, m_nodes);
                }
            }

            // Add children to stack if we're not
            if (!m_nodes[sNode.index].isLeaf) {
                const STreeNode& node = m_nodes[sNode.index];
                for (int i = 0; i < 2; ++i) {
                    nodeIndices.push({node.children[i], sNode.depth + 1});
                }
            }
        }

        // Uncomment once memory becomes an issue.
        //m_nodes.shrink_to_fit();
    }

private:
    std::vector<STreeNode> m_nodes;
    AABB m_aabb;
};


static StatsCounter avgPathLength("Guided path tracer", "Average path length", EAverage);

class GuidedPathTracer : public MonteCarloIntegrator {
public:
    GuidedPathTracer(const Properties &props) : MonteCarloIntegrator(props) {
        m_neeStr = props.getString("nee", "never");
        if (m_neeStr == "never") {
            m_nee = ENever;
        } else if (m_neeStr == "kickstart") {
            m_nee = EKickstart;
        } else if (m_neeStr == "always") {
            m_nee = EAlways;
        } else {
            Assert(false);
        }

        m_sdTreeMaxMemory = props.getInteger("sdTreeMaxMemory", -1);
        m_sTreeThreshold = props.getInteger("sTreeThreshold", 12000);
        m_dTreeThreshold = props.getFloat("dTreeThreshold", 0.01f);
        m_bsdfSamplingFraction = props.getFloat("bsdfSamplingFraction", 0.5f);
        m_automaticBudget = props.getBoolean("automaticBudget", true);
        m_sppPerPass = props.getInteger("sppPerPass", 4);

        m_budgetStr = props.getString("budgetType", "seconds");
        if (m_budgetStr == "spp") {
            m_budgetType = ESpp;
        } else if (m_budgetStr == "seconds") {
            m_budgetType = ESeconds;
        } else {
            Assert(false);
        }

        m_budget = props.getFloat("budget", 300.0f);
        m_dumpSDTree = props.getBoolean("dumpSDTree", false);
    }

    ref<BlockedRenderProcess> renderPass(Scene *scene,
        RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        /* This is a sampling-based integrator - parallelize */
        ref<BlockedRenderProcess> proc = new BlockedRenderProcess(job,
            queue, scene->getBlockSize());

        proc->disableProgress();

        proc->bindResource("integrator", integratorResID);
        proc->bindResource("scene", sceneResID);
        proc->bindResource("sensor", sensorResID);
        proc->bindResource("sampler", samplerResID);

        scene->bindUsedResources(proc);
        bindUsedResources(proc);

        return proc;
    }

    void resetSDTree() {
        Log(EInfo, "Resetting distributions for sampling.");

        m_sdTree->refine((size_t)(std::pow(2, m_iter / 2.0f) * m_sTreeThreshold), m_sdTreeMaxMemory);
        m_sdTree->forEachDTreeWrapperParallel([this](DTreeWrapper* dTree) { dTree->reset(20, m_dTreeThreshold); });
    }

    void buildSDTree() {
        Log(EInfo, "Building distributions for sampling.");

        // Build distributions
        m_sdTree->forEachDTreeWrapperParallel([](DTreeWrapper* dTree) { dTree->build(); });

        // Gather statistics
        int maxDepth = 0;
        int minDepth = std::numeric_limits<int>::max();
        Float avgDepth = 0;
        Float maxAvgRadiance = 0;
        Float minAvgRadiance = std::numeric_limits<Float>::max();
        Float avgAvgRadiance = 0;
        size_t maxNodes = 0;
        size_t minNodes = std::numeric_limits<size_t>::max();
        Float avgNodes = 0;
        size_t maxSamples = 0;
        size_t minSamples = std::numeric_limits<size_t>::max();
        Float avgSamples = 0;

        int nPoints = 0;
        int nPointsNodes = 0;

        m_sdTree->forEachDTreeWrapperConst([&](const DTreeWrapper* dTree) {
            const int depth = dTree->depth();
            maxDepth = std::max(maxDepth, depth);
            minDepth = std::min(minDepth, depth);
            avgDepth += depth;

            const Float avgRadiance = dTree->meanRadiance();
            maxAvgRadiance = std::max(maxAvgRadiance, avgRadiance);
            minAvgRadiance = std::min(minAvgRadiance, avgRadiance);
            avgAvgRadiance += avgRadiance;

            if (dTree->amountNodes() > 1) {
                const size_t nodes = dTree->amountNodes();
                maxNodes = std::max(maxNodes, nodes);
                minNodes = std::min(minNodes, nodes);
                avgNodes += nodes;
                ++nPointsNodes;
            }

            const size_t samples = dTree->amountSamples();
            maxSamples = std::max(maxSamples, samples);
            minSamples = std::min(minSamples, samples);
            avgSamples += samples;

            ++nPoints;
        });

        if (nPoints > 0) {
            avgDepth /= nPoints;
            avgAvgRadiance /= nPoints;

            if (nPointsNodes > 0) {
                avgNodes /= nPointsNodes;
            }

            avgSamples /= nPoints;
        }

        Log(EInfo,
            "Distribution statistics:\n"
            "  Depth          = [%d, %f, %d]\n"
            "  Mean radiance  = [%f, %f, %f]\n"
            "  Amount nodes   = [" SIZE_T_FMT ", %f, " SIZE_T_FMT "]\n"
            "  Amount samples = [" SIZE_T_FMT ", %f, " SIZE_T_FMT "]\n",
            minDepth, avgDepth, maxDepth,
            minAvgRadiance, avgAvgRadiance, maxAvgRadiance,
            minNodes, avgNodes, maxNodes,
            minSamples, avgSamples, maxSamples
        );

        m_isBuilt = true;
    }

    void dumpSDTree(Scene* scene, ref<Sensor> sensor) {
        std::ostringstream extension;
        extension << "-" << std::setfill('0') << std::setw(2) << m_iter << ".sdt";
        fs::path path = scene->getDestinationFile();
        path = path.parent_path() / (path.leaf().string() + extension.str());

        auto cameraMatrix = sensor->getWorldTransform()->eval(0).getMatrix();

        BlobWriter blob(path.string());

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                blob << (float)cameraMatrix(i, j);
            }
        }

        m_sdTree->dump(blob);
    }

    bool performRenderPasses(Float& variance, int numPasses, Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        m_image->clear();
        m_squaredImage->clear();

        size_t totalBlocks = 0;

        Log(EInfo, "Rendering %d render passes.", numPasses);

        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < numPasses; ++i) {
            ref<BlockedRenderProcess> process = renderPass(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
            m_renderProcesses.push_back(process);
            totalBlocks += process->totalBlocks();
        }

        bool result = true;
        int passesRenderedLocal = 0;

        static const size_t processBatchSize = 128;

        for (size_t i = 0; i < m_renderProcesses.size(); i += processBatchSize) {
            const size_t start = i;
            const size_t end = std::min(i + processBatchSize, m_renderProcesses.size());
            for (size_t j = start; j < end; ++j) {
                sched->schedule(m_renderProcesses[j]);
            }

            for (size_t j = start; j < end; ++j) {
                auto& process = m_renderProcesses[j];
                sched->wait(process);

                ++m_passesRendered;
                ++m_passesRenderedThisIter;
                ++passesRenderedLocal;

                int progress = 0;
                bool shouldAbort;
                switch (m_budgetType) {
                    case ESpp:
                        progress = m_passesRendered;
                        shouldAbort = false;
                        break;
                    case ESeconds:
                        progress = (int)computeElapsedSeconds(m_startTime);
                        shouldAbort = progress > m_budget;
                        break;
                    default:
                        Assert(false);
                        break;
                }

                m_progress->update(progress);

                if (process->getReturnStatus() != ParallelProcess::ESuccess) {
                    result = false;
                    shouldAbort = true;
                }

                if (shouldAbort) {
                    goto l_abort;
                }
            }
        }
    l_abort:

        for (auto& process : m_renderProcesses) {
            sched->cancel(process);
        }

        m_renderProcesses.clear();

        variance = 0;
        Bitmap* squaredImage = m_squaredImage->getBitmap();
        Bitmap* image = m_image->getBitmap();

        m_varianceBuffer->clear();

        int N = passesRenderedLocal * m_sppPerPass;

        Vector2i size = squaredImage->getSize();
        for (int x = 0; x < size.x; ++x)
            for (int y = 0; y < size.y; ++y) {
                Point2i pos = Point2i(x, y);
                Spectrum pixel = image->getPixel(pos);
                Spectrum localVar = squaredImage->getPixel(pos) - pixel * pixel / (Float)N;
                image->setPixel(pos, localVar);
                variance += localVar.getLuminance();
            }

        variance /= (Float)size.x * size.y * (N - 1);
        m_varianceBuffer->put(m_image);

        Float seconds = computeElapsedSeconds(start);

        const Float ttuv = seconds * variance;
        const Float stuv = passesRenderedLocal * 4 * variance;
        Log(EInfo, "%.2f seconds, Total passes: %d, Var: %f, TTUV: %f, STUV: %f.",
            seconds, m_passesRendered, variance, ttuv, stuv);

        return result;
    }

    bool doNeeWithSpp(int spp) {
        switch (m_nee) {
            case ENever:
                return false;
            case EKickstart:
                return spp < 128;
            default:
                return true;
        }
    }

    bool renderSPP(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        ref<Scheduler> sched = Scheduler::getInstance();

        size_t sampleCount = (size_t)m_budget;

        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        int nPasses = (int)std::ceil(sampleCount / (Float)m_sppPerPass);
        sampleCount = m_sppPerPass * nPasses;

        bool result = true;
        Float currentVarAtEnd = std::numeric_limits<Float>::infinity();

        m_progress = std::unique_ptr<ProgressReporter>(new ProgressReporter("Rendering", nPasses, job));

        while (result && m_passesRendered < nPasses) {
            const int sppRendered = m_passesRendered * m_sppPerPass;
            m_doNee = doNeeWithSpp(sppRendered);

            int remainingPasses = nPasses - m_passesRendered;
            const int passesThisIteration = std::min(remainingPasses, 1 << m_iter);

            Log(EInfo, "ITERATION %d, %d passes", m_iter, passesThisIteration);

            film->clear();
            resetSDTree();

            Float variance;
            if (!performRenderPasses(variance, passesThisIteration, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID)) {
                result = false;
                break;
            }

            const Float lastVarAtEnd = currentVarAtEnd;
            currentVarAtEnd = passesThisIteration * variance / remainingPasses;

            Log(EInfo,
                "Extrapolated var:\n"
                "  Last:    %f\n"
                "  Current: %f\n",
                lastVarAtEnd, currentVarAtEnd);

            remainingPasses -= passesThisIteration;
            if (m_automaticBudget && remainingPasses > 0 && (
                    // if there is any time remaining we want to keep going if
                    // either will have less time next iter
                    remainingPasses < passesThisIteration ||
                    // or, according to the convergence behavior, we're better off if we keep going
                    // (we only trust the variance if we drew enough samples for it to be a reliable estimate,
                    // captured by an arbitraty threshold).
                    (sppRendered > 256 && currentVarAtEnd > lastVarAtEnd)
                )) {
                Log(EInfo, "FINAL %d passes", remainingPasses);
                if (!performRenderPasses(variance, remainingPasses, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID)) {
                    result = false;
                    break;
                }
            }
            buildSDTree();

            if (m_dumpSDTree) {
                dumpSDTree(scene, sensor);
            }

            ++m_iter;
            m_passesRenderedThisIter = 0;
        }

        return result;
    }

    static Float computeElapsedSeconds(std::chrono::steady_clock::time_point start) {
        auto current = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
        return (Float)ms.count() / 1000;
    }

    bool renderTime(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        Float nSeconds = m_budget;

        bool result = true;
        Float currentVarAtEnd = std::numeric_limits<Float>::infinity();

        m_progress = std::unique_ptr<ProgressReporter>(new ProgressReporter("Rendering", (int)nSeconds, job));

        Float elapsedSeconds = 0;

        while (result && elapsedSeconds < nSeconds) {
            const int sppRendered = m_passesRendered * m_sppPerPass;
            m_doNee = doNeeWithSpp(sppRendered);

            Float remainingTime = nSeconds - elapsedSeconds;
            const int passesThisIteration = 1 << m_iter;

            Log(EInfo, "ITERATION %d, %d passes", m_iter, passesThisIteration);

            const auto startIter = std::chrono::steady_clock::now();

            film->clear();
            resetSDTree();

            Float variance;
            if (!performRenderPasses(variance, passesThisIteration, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID)) {
                result = false;
                break;
            }

            const Float secondsIter = computeElapsedSeconds(startIter);

            const Float lastVarAtEnd = currentVarAtEnd;
            currentVarAtEnd = secondsIter * variance / remainingTime;

            Log(EInfo,
                "Extrapolated var:\n"
                "  Last:    %f\n"
                "  Current: %f\n",
                lastVarAtEnd, currentVarAtEnd);

            remainingTime -= secondsIter;
            if (m_automaticBudget && remainingTime > 0 && (
                    // if there is any time remaining we want to keep going if
                    // either will have less time next iter
                    remainingTime < secondsIter ||
                    // or, according to the convergence behavior, we're better off if we keep going
                    // (we only trust the variance if we drew enough samples for it to be a reliable estimate,
                    // captured by an arbitraty threshold).
                    (sppRendered > 256 && currentVarAtEnd > lastVarAtEnd)
                )) {
                Log(EInfo, "FINAL %f seconds", remainingTime);
                do {
                    if (!performRenderPasses(variance, passesThisIteration, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID)) {
                        result = false;
                        break;
                    }

                    elapsedSeconds = computeElapsedSeconds(m_startTime);
                } while (elapsedSeconds < nSeconds);
            }
            buildSDTree();

            if (m_dumpSDTree) {
                dumpSDTree(scene, sensor);
            }

            ++m_iter;
            m_passesRenderedThisIter = 0;
            elapsedSeconds = computeElapsedSeconds(m_startTime);
        }

        return result;
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID) override {

        m_sdTree = std::unique_ptr<STree>(new STree(scene->getAABB()));
        m_iter = 0;

        ref<Scheduler> sched = Scheduler::getInstance();

        size_t nCores = sched->getCoreCount();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        auto properties = Properties("hdrfilm");
        properties.setInteger("width", film->getSize().x);
        properties.setInteger("height", film->getSize().y);
        m_varianceBuffer = static_cast<Film*>(PluginManager::getInstance()->createObject(MTS_CLASS(Film), properties));
        m_varianceBuffer->setDestinationFile(scene->getDestinationFile(), 0);

        m_squaredImage = new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getSize(), film->getReconstructionFilter());
        m_image = new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getSize(), film->getReconstructionFilter());

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y, nCores, nCores == 1 ? "core" : "cores");

        Thread::initializeOpenMP(nCores);

        int integratorResID = sched->registerResource(this);
        bool result = true;

        m_startTime = std::chrono::steady_clock::now();

        m_passesRendered = 0;
        switch (m_budgetType) {
            case ESpp:
                renderSPP(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
                break;
            case ESeconds:
                renderTime(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
                break;
            default:
                Assert(false);
                break;
        }

        sched->unregisterResource(integratorResID);

        m_progress = nullptr;

        return result;
    }

    void renderBlock(const Scene *scene, const Sensor *sensor,
        Sampler *sampler, ImageBlock *block, const bool &stop,
        const std::vector< TPoint2<uint8_t> > &points) const override {

        Float diffScaleFactor = 1.0f /
            std::sqrt((Float)m_sppPerPass);

        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        RadianceQueryRecord rRec(scene, sampler);
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        RayDifferential sensorRay;

        block->clear();

        ref<ImageBlock> squaredBlock = new ImageBlock(block->getPixelFormat(), block->getSize(), block->getReconstructionFilter());
        squaredBlock->setOffset(block->getOffset());
        squaredBlock->clear();

        uint32_t queryType = RadianceQueryRecord::ESensorRay;

        if (!sensor->getFilm()->hasAlpha()) // Don't compute an alpha channel if we don't have to
            queryType &= ~RadianceQueryRecord::EOpacity;

        for (size_t i = 0; i < points.size(); ++i) {
            Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset());
            if (stop)
                break;

            for (int j = 0; j < m_sppPerPass; j++) {
                rRec.newQuery(queryType, sensor->getMedium());
                Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

                if (needsApertureSample)
                    apertureSample = rRec.nextSample2D();
                if (needsTimeSample)
                    timeSample = rRec.nextSample1D();

                Spectrum spec = sensor->sampleRayDifferential(
                    sensorRay, samplePos, apertureSample, timeSample);

                sensorRay.scaleDifferential(diffScaleFactor);

                spec *= Li(sensorRay, rRec);
                block->put(samplePos, spec, rRec.alpha);
                squaredBlock->put(samplePos, spec * spec, rRec.alpha);
                sampler->advance();
            }
        }

        m_squaredImage->put(squaredBlock);
        m_image->put(block);
    }

    void cancel() {
        const auto& scheduler = Scheduler::getInstance();
        for (size_t i = 0; i < m_renderProcesses.size(); ++i) {
            scheduler->cancel(m_renderProcesses[i]);
        }
    }

    Spectrum sampleMat(const BSDF* bsdf, BSDFSamplingRecord& bRec, Float& pdf, RadianceQueryRecord& rRec, const DTreeWrapper* dTree) const {
        Point2 sample = rRec.nextSample2D();

        auto type = bsdf->getType();
        if (!m_isBuilt || !dTree || (type & BSDF::EDelta) == (type & BSDF::EAll)) {
            return bsdf->sample(bRec, pdf, sample);
        }

        Spectrum result;
        if (sample.x < m_bsdfSamplingFraction) {
            sample.x /= m_bsdfSamplingFraction;
            result = bsdf->sample(bRec, pdf, sample);
            if (result.isZero()) {
                return Spectrum{0.0f};
            }

            // If we sampled a delta component, then we have a 0 probability
            // of sampling that direction via guiding, thus we can return early.
            if (bRec.sampledType & BSDF::EDelta) {
                pdf *= m_bsdfSamplingFraction;
                return result / m_bsdfSamplingFraction;
            }

            result *= pdf;
        } else {
            sample.x = (sample.x - m_bsdfSamplingFraction) / (1 - m_bsdfSamplingFraction);
            bRec.wo = bRec.its.toLocal(dTree->sampleDirection(sample));
            result = bsdf->eval(bRec);
        }

        pdf = pdfMat(bsdf, bRec, dTree);
        if (pdf == 0) {
            return Spectrum{0.0f};
        }

        return result / pdf;
    }

    Float pdfMat(const BSDF* bsdf, const BSDFSamplingRecord& bRec, const DTreeWrapper* dTree) const {
        auto type = bsdf->getType();
        if (!m_isBuilt || !dTree || (type & BSDF::EDelta) == (type & BSDF::EAll)) {
            return bsdf->pdf(bRec);
        }

        const Float bsdfPdf = bsdf->pdf(bRec);
        if (bsdfPdf <= 0 || !std::isfinite(bsdfPdf)) {
            return 0;
        }

        return
            m_bsdfSamplingFraction * bsdfPdf +
            (1 - m_bsdfSamplingFraction) * dTree->samplePdf(bRec.its.toWorld(bRec.wo));
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        struct Vertex {
            DTreeWrapper* dTree;
            Ray ray;
            Spectrum throughput;
            Spectrum radiance;

            void record(Spectrum& r) {
                if (throughput[0] > 0) radiance[0] += r[0] / throughput[0];
                if (throughput[1] > 0) radiance[1] += r[1] / throughput[1];
                if (throughput[2] > 0) radiance[2] += r[2] / throughput[2];
            }

            void recordMeasurement(Spectrum& r) {
                dTree->recordMeasurement(r);
            }

            void commit(Point2 sample, bool shouldCount) {
                dTree->recordRadiance(ray.o, ray.d, radiance, sample, shouldCount);
            }
        };

        static const int AMOUNT_VERTICES = 32;
        std::array<Vertex, AMOUNT_VERTICES> vertices;

        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        MediumSamplingRecord mRec;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        Float eta = 1.0f;

        /* Perform the first ray intersection (or ignore if the
        intersection has already been provided). */
        rRec.rayIntersect(ray);

        Spectrum throughput(1.0f);
        Spectrum measurementEstimate;
        bool scattered = false;

        int depth = 0;

        auto recordRadiance = [&](Spectrum radiance) {
            Li += radiance;
            for (int i = 0; i < depth; ++i) {
                vertices[i].record(radiance);
            }
        };

#ifdef PATH_DEBUG
        std::string pathString = "E";
#endif

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {

            /* ==================================================================== */
            /*                 Radiative Transfer Equation sampling                 */
            /* ==================================================================== */
            if (rRec.medium && rRec.medium->sampleDistance(Ray(ray, 0, its.t), mRec, rRec.sampler)) {
                /* Sample the integral
                \int_x^y tau(x, x') [ \sigma_s \int_{S^2} \rho(\omega,\omega') L(x,\omega') d\omega' ] dx'
                */
                const PhaseFunction *phase = mRec.getPhaseFunction();

                if (rRec.depth >= m_maxDepth && m_maxDepth != -1) // No more scattering events allowed
                    break;

                throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;

                /* ==================================================================== */
                /*                          Luminaire sampling                          */
                /* ==================================================================== */

                /* Estimate the single scattering component if this is requested */
                DirectSamplingRecord dRec(mRec.p, mRec.time);

                if (rRec.type & RadianceQueryRecord::EDirectMediumRadiance) {
                    int interactions = m_maxDepth - rRec.depth - 1;

                    Spectrum value = scene->sampleAttenuatedEmitterDirect(
                        dRec, rRec.medium, interactions,
                        rRec.nextSample2D(), rRec.sampler);

                    if (!value.isZero()) {
                        const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                        /* Evaluate the phase function */
                        PhaseFunctionSamplingRecord pRec(mRec, -ray.d, dRec.d);
                        Float phaseVal = phase->eval(pRec);

                        if (phaseVal != 0) {
                            /* Calculate prob. of having sampled that direction using
                            phase function sampling */
                            Float phasePdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                                ? phase->pdf(pRec) : (Float) 0.0f;

                            /* Weight using the power heuristic */
                            const Float weight = miWeight(dRec.pdf, phasePdf);
                            recordRadiance(throughput * value * phaseVal * weight);
                        }
                    }
                }

                /* ==================================================================== */
                /*                         Phase function sampling                      */
                /* ==================================================================== */

                Float phasePdf;
                PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
                Float phaseVal = phase->sample(pRec, phasePdf, rRec.sampler);
                if (phaseVal == 0)
                    break;
                throughput *= phaseVal;

                /* Trace a ray in this direction */
                ray = Ray(mRec.p, pRec.wo, ray.time);
                ray.mint = 0;

                Spectrum value(0.0f);
                rayIntersectAndLookForEmitter(scene, rRec.sampler, rRec.medium,
                    m_maxDepth - rRec.depth - 1, ray, its, dRec, value);

                /* If a luminaire was hit, estimate the local illumination and
                weight using the power heuristic */
                if (!value.isZero() && (rRec.type & RadianceQueryRecord::EDirectMediumRadiance)) {
                    const Float emitterPdf = scene->pdfEmitterDirect(dRec);
                    recordRadiance(throughput * value * miWeight(phasePdf, emitterPdf));
                }

                /* ==================================================================== */
                /*                         Multiple scattering                          */
                /* ==================================================================== */

                /* Stop if multiple scattering was not requested */
                if (!(rRec.type & RadianceQueryRecord::EIndirectMediumRadiance))
                    break;
                rRec.type = RadianceQueryRecord::ERadianceNoEmission;
            } else {
                /* Sample
                tau(x, y) (Surface integral). This happens with probability mRec.pdfFailure
                Account for this and multiply by the proper per-color-channel transmittance.
                */
                if (rRec.medium)
                    throughput *= mRec.transmittance / mRec.pdfFailure;

                if (!its.isValid()) {
                    /* If no intersection could be found, possibly return
                    attenuated radiance from a background luminaire */
                    if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                        && (!m_hideEmitters || scattered)) {
                        Spectrum value = throughput * scene->evalEnvironment(ray);
                        if (rRec.medium)
                            value *= rRec.medium->evalTransmittance(ray, rRec.sampler);
                        recordRadiance(value);
                    }

                    break;
                }

                /* Possibly include emitted radiance if requested */
                if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                    recordRadiance(throughput * its.Le(-ray.d));

                /* Include radiance from a subsurface integrator if requested */
                if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                    recordRadiance(throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth));

                if (rRec.depth >= m_maxDepth && m_maxDepth != -1)
                    break;

                /* Prevent light leaks due to the use of shading normals */
                Float wiDotGeoN = -dot(its.geoFrame.n, ray.d),
                    wiDotShN = Frame::cosTheta(its.wi);
                if (wiDotGeoN * wiDotShN < 0 && m_strictNormals)
                    break;

                const BSDF *bsdf = its.getBSDF(ray);

                DTreeWrapper* dTree = nullptr;

                // We only guide smooth BRDFs for now. Analytic product sampling
                // would be conceivable for discrete decisions such as refraction vs
                // reflection.
                if (bsdf->getType() & BSDF::ESmooth) {
                    dTree = m_sdTree->dTreeWrapper(its.p);
                }

                /* ==================================================================== */
                /*                            BSDF sampling                             */
                /* ==================================================================== */

                /* Sample BSDF * cos(theta) */
                BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
                Float bsdfPdf;
                Spectrum bsdfWeight = sampleMat(bsdf, bRec, bsdfPdf, rRec, dTree);

                /* ==================================================================== */
                /*                          Luminaire sampling                          */
                /* ==================================================================== */

                DirectSamplingRecord dRec(its);

                /* Estimate the direct illumination if this is requested */
                if (m_doNee &&
                    (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
                    (bsdf->getType() & BSDF::ESmooth)) {
                    int interactions = m_maxDepth - rRec.depth - 1;

                    Spectrum value = scene->sampleAttenuatedEmitterDirect(
                        dRec, its, rRec.medium, interactions,
                        rRec.nextSample2D(), rRec.sampler);

                    if (!value.isZero()) {

                        BSDFSamplingRecord bRec(its, its.toLocal(dRec.d));

                        Float woDotGeoN = dot(its.geoFrame.n, dRec.d);

                        /* Prevent light leaks due to the use of shading normals */
                        if (!m_strictNormals || woDotGeoN * Frame::cosTheta(bRec.wo) > 0) {

                            /* Evaluate BSDF * cos(theta) */
                            const Spectrum bsdfVal = bsdf->eval(bRec);

                            /* Calculate prob. of having generated that direction using BSDF sampling */
                            const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
                            Float bsdfPdf = (emitter->isOnSurface()
                                && dRec.measure == ESolidAngle)
                                ? pdfMat(bsdf, bRec, dTree) : (Float) 0.0f;

                            /* Weight using the power heuristic */
                            const Float weight = miWeight(dRec.pdf, bsdfPdf);

                            if (dTree) {
                                Vertex v = Vertex{
                                    dTree,
                                    Ray(its.p, dRec.d, 0),
                                    Spectrum{0.0f},
                                    value * weight,
                                };

                                v.commit(rRec.nextSample2D(), false);
                            }

                            value *= bsdfVal;
                            recordRadiance(throughput * value * weight);
                        }
                    }
                }

                // BSDF handling
                if (bsdfWeight.isZero())
                    break;

                /* Prevent light leaks due to the use of shading normals */
                const Vector wo = its.toWorld(bRec.wo);
                Float woDotGeoN = dot(its.geoFrame.n, wo);

                if (woDotGeoN * Frame::cosTheta(bRec.wo) <= 0 && m_strictNormals)
                    break;

                /* Trace a ray in this direction */
                ray = Ray(its.p, wo, ray.time);

                /* Keep track of the throughput, medium, and relative
                refractive index along the path */
                throughput *= bsdfWeight;
                eta *= bRec.eta;
                if (its.isMediumTransition())
                    rRec.medium = its.getTargetMedium(ray.d);

                Spectrum value(0.0f);
                rayIntersectAndLookForEmitter(scene, rRec.sampler, rRec.medium,
                    m_maxDepth - rRec.depth - 1, ray, its, dRec, value);

                /* If a luminaire was hit, estimate the local illumination and
                weight using the power heuristic */
                if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) {
                    const Float emitterPdf = (m_doNee && !(bRec.sampledType & BSDF::EDelta) && !value.isZero()) ?
                        scene->pdfEmitterDirect(dRec) : 0;

                    const Float weight = miWeight(bsdfPdf, emitterPdf);
                    if (!value.isZero()) {
                        recordRadiance(throughput * value * weight);
                    }

                    if (!(bRec.sampledType & BSDF::EDelta) && dTree && depth < AMOUNT_VERTICES) {
                        if (depth == 0 && m_isBuilt) {
                            measurementEstimate = dTree->measurementEstimate();
                        }

                        Float factor = 1 / bsdfPdf;
                        if (factor > 0) {
                            vertices[depth] = Vertex{
                                dTree,
                                ray,
                                throughput / factor,
                                value * factor * weight,
                            };

                            ++depth;
                        }
                    }
                }

                /* ==================================================================== */
                /*                         Indirect illumination                        */
                /* ==================================================================== */

                /* Stop if indirect illumination was not requested */
                if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                    break;

                rRec.type = RadianceQueryRecord::ERadianceNoEmission;

                // Russian roulette
                if (rRec.depth++ >= m_rrDepth) {
                    Float successProb = 1.0f;
                    if (dTree && !(bRec.sampledType & BSDF::EDelta)) {
                        if (!m_isBuilt) {
                            successProb = throughput.max() * eta * eta;
                        } else {
                            // Adjoint-based russian roulette based on Vorba and Křivánek [2016]
                            Spectrum incidentRadiance = Spectrum{dTree->estimateRadiance(ray.d)};

                            if (measurementEstimate.min() > 0 && incidentRadiance.min() > 0) {
                                const Float center = (measurementEstimate / incidentRadiance).average();
                                const Float s = 5;
                                const Float wMin = 2 * center / (1 + s);
                                //const Float wMax = wMin * 5; // Useful for splitting

                                const Float tMax = throughput.max();
                                //const Float tMin = throughput.min(); // Useful for splitting

                                // Splitting is not supported.
                                if (tMax < wMin) {
                                    successProb = wMin / tMax;
                                }
                            }
                        }

                        successProb = std::max(0.1f, std::min(successProb, 0.99f));
                    }

                    if (rRec.nextSample1D() >= successProb)
                        break;
                    throughput /= successProb;
                }
            }

            scattered = true;
        }
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        if (depth > 0) {
            const Point2 sample = rRec.nextSample2D();
            vertices[0].recordMeasurement(Li);
            for (int i = 0; i < depth; ++i) {
                vertices[i].commit(sample, true);
            }
        }

        return Li;
    }

    /**
    * This function is called by the recursive ray tracing above after
    * having sampled a direction from a BSDF/phase function. Due to the
    * way in which this integrator deals with index-matched boundaries,
    * it is necessarily a bit complicated (though the improved performance
    * easily pays for the extra effort).
    *
    * This function
    *
    * 1. Intersects 'ray' against the scene geometry and returns the
    *    *first* intersection via the '_its' argument.
    *
    * 2. It checks whether the intersected shape was an emitter, or if
    *    the ray intersects nothing and there is an environment emitter.
    *    In this case, it returns the attenuated emittance, as well as
    *    a DirectSamplingRecord that can be used to query the hypothetical
    *    sampling density at the emitter.
    *
    * 3. If current shape is an index-matched medium transition, the
    *    integrator keeps on looking on whether a light source eventually
    *    follows after a potential chain of index-matched medium transitions,
    *    while respecting the specified 'maxDepth' limits. It then returns
    *    the attenuated emittance of this light source, while accounting for
    *    all attenuation that occurs on the wya.
    */
    void rayIntersectAndLookForEmitter(const Scene *scene, Sampler *sampler,
        const Medium *medium, int maxInteractions, Ray ray, Intersection &_its,
        DirectSamplingRecord &dRec, Spectrum &value) const {
        Intersection its2, *its = &_its;
        Spectrum transmittance(1.0f);
        bool surface = false;
        int interactions = 0;

        while (true) {
            surface = scene->rayIntersect(ray, *its);

            if (medium)
                transmittance *= medium->evalTransmittance(Ray(ray, 0, its->t), sampler);

            if (surface && (interactions == maxInteractions ||
                !(its->getBSDF()->getType() & BSDF::ENull) ||
                its->isEmitter())) {
                /* Encountered an occluder / light source */
                break;
            }

            if (!surface)
                break;

            if (transmittance.isZero())
                return;

            if (its->isMediumTransition())
                medium = its->getTargetMedium(ray.d);

            Vector wo = its->shFrame.toLocal(ray.d);
            BSDFSamplingRecord bRec(*its, -wo, wo, ERadiance);
            bRec.typeMask = BSDF::ENull;
            transmittance *= its->getBSDF()->eval(bRec, EDiscrete);

            ray.o = ray(its->t);
            ray.mint = Epsilon;
            its = &its2;

            if (++interactions > 100) { /// Just a precaution..
                Log(EWarn, "rayIntersectAndLookForEmitter(): round-off error issues?");
                return;
            }
        }

        if (surface) {
            /* Intersected something - check if it was a luminaire */
            if (its->isEmitter()) {
                dRec.setQuery(ray, *its);
                value = transmittance * its->Le(-ray.d);
            }
        } else {
            /* Intersected nothing -- perhaps there is an environment map? */
            const Emitter *env = scene->getEnvironmentEmitter();

            if (env && env->fillDirectSamplingRecord(dRec, ray)) {
                value = transmittance * env->evalEnvironment(RayDifferential(ray));
                dRec.dist = std::numeric_limits<Float>::infinity();
                its->t = std::numeric_limits<Float>::infinity();
            }
        }
    }

    Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA; pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "GuidedPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

private:
    /// The datastructure for guiding paths.
    std::unique_ptr<STree> m_sdTree;

    /// The squared values of our currently rendered image. Used to estimate variance.
    mutable ref<ImageBlock> m_squaredImage;
    /// The currently rendered image. Used to estimate variance.
    mutable ref<ImageBlock> m_image;

    /// This contains the currently estimated variance.
    mutable ref<Film> m_varianceBuffer;

    /// The modes of NEE which are supported.
    enum ENee {
        ENever,
        EKickstart,
        EAlways,
    };

    /**
        How to perform next event estimation (NEE). The following values are valid:
        - "never":     Never performs NEE.
        - "kickstart": Performs NEE for the first few iterations to initialize
                       the SDTree with good direct illumination estimates.
        - "always":    Always performs NEE.
        Default = "never"
    */
    std::string m_neeStr;
    ENee m_nee;

    /// Whether Li should currently perform NEE (automatically set during rendering based on m_nee).
    bool m_doNee;

    enum EBudget {
        ESpp,
        ESeconds,
    };

    /**
        What type of budget to use. The following values are valid:
        - "spp":     Budget is the amount of samples per pixel.
        - "seconds": Budget is a time in seconds.
        Default = "seconds"
    */
    std::string m_budgetStr;
    EBudget m_budgetType;
    Float m_budget;

    bool m_isBuilt = false;
    int m_iter;

    int m_sppPerPass;

    int m_passesRendered;
    int m_passesRenderedThisIter;
    mutable std::unique_ptr<ProgressReporter> m_progress;

    std::vector<ref<BlockedRenderProcess>> m_renderProcesses;

    /// Whether to automatically stop training when optimal. If off, then iterations are performed until the budget is exhausted.
    bool m_automaticBudget;

    /// Maximum memory footprint of the SDTree in MB. Stops subdividing once reached. -1 to disable.
    int m_sdTreeMaxMemory;

    /**
        Leaf nodes of the spatial binary tree are subdivided if the amount of samples
        they received in the last iteration exceeds c * sqrt(2^k) where c is this value
        and k is the iteration index. The first iteration has k==0.
        Default = 12000
    */
    int m_sTreeThreshold;

    /**
        Leaf nodes of the directional quadtree are subdivided if the fraction
        of energy they carry exceeds this value.
        Default = 0.01 (1%)
    */
    Float m_dTreeThreshold;

    /**
        When guiding, we perform MIS with the balance heuristic between the guiding
        distribution and the BSDF, combined with probabilistically choosing one of the
        two sampling methods. This factor controls how often the BSDF is sampled
        vs. how often the guiding distribution is sampled.
        Default = 0.5 (50%)
    */
    Float m_bsdfSamplingFraction;

    /**
        Whether to dump a binary representation of the SD-Tree to disk after every
        iteration. The dumped SD-Tree can be visualized with the accompanying
        visualizer tool.
        Default = false
    */
    bool m_dumpSDTree;

    /// The time at which rendering started.
    std::chrono::steady_clock::time_point m_startTime;

public:
    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS(GuidedPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(GuidedPathTracer, "Guided path tracer");
MTS_NAMESPACE_END
