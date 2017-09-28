/*
    This file was adapted from nanogui/src/example1.cpp which is
    governed by a BSD-style license that can be found in the
    nanogui/LICENSE.txt file.

    Copyright (c) 2017 by ETH Zurich, Thomas Mueller.
*/

#include <nanogui/opengl.h>
#include <nanogui/glutil.h>
#include <nanogui/screen.h>
#include <nanogui/window.h>
#include <nanogui/layout.h>
#include <nanogui/label.h>
#include <nanogui/button.h>
#include <nanogui/entypo.h>
#include <nanogui/textbox.h>
#include <nanogui/slider.h>
#include <nanogui/imageview.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <iomanip> // setprecision
#include <sstream> // stringstream
#include <memory>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace Eigen;
using namespace nanogui;
using namespace std;

class BlobReader {
public:
    BlobReader(const string& filename) : f(filename, ios::in | ios::binary) {}

    template <typename Type>
    typename enable_if<is_standard_layout<Type>::value, BlobReader&>::type
        operator >> (Type& Element) {
        Read(&Element, 1);
        return *this;
    }

    // CAUTION: This function may break down on big-endian architectures.
    //          The ordering of bytes has to be reverted then.
    template <typename T>
    void Read(T* Dest, size_t Size) {
        f.read(reinterpret_cast<char*>(Dest), Size * sizeof(T));
    }

    bool isValid() const {
        return (bool)(f);
    }

private:
    ifstream f;
};

static const int NUM_CHANNELS = 1;

struct QuadTreeNode {
    array<array<float, 4>, NUM_CHANNELS> data;
    array<uint16_t, 4> children;

    inline bool isLeaf(int index) const {
        return children[index] == 0;
    }

    int computeDepth(const vector<QuadTreeNode>& nodes) const {
        int maxDepth = 0;
        for (int i = 0; i < 4; ++i) {
            if (!isLeaf(i)) {
                maxDepth = max(maxDepth, nodes[children[i]].computeDepth(nodes) + 1);
            }
        }

        return maxDepth;
    }

    float computeMax(const vector<QuadTreeNode>& nodes) const {
        float maximum = 0;
        for (int i = 0; i < 4; ++i) {
            if (!isLeaf(i)) {
                maximum = max(maximum, nodes[children[i]].computeMax(nodes));
            } else {
                maximum = max(maximum, data[0][i]);
            }
        }

        return 4 * maximum;
    }

    int getChildIndex(Vector2f& p) const {
        if (p.x() < 0.5f) {
            p.x() *= 2;
            if (p.y() < 0.5f) {
                p.y() *= 2;
                return 0;
            } else {
                p.y() = (p.y() - 0.5f) * 2;
                return 1;
            }
        } else {
            p.x() = (p.x() - 0.5f) * 2;
            if (p.y() < 0.5f) {
                p.y() *= 2;
                return 2;
            } else {
                p.y() = (p.y() - 0.5f) * 2;
                return 3;
            }
        }
    }

    float eval(int index, Vector2f& p, const vector<QuadTreeNode>& nodes) const {
        const int c = getChildIndex(p);
        if (isLeaf(c)) {
            return data[index][c];
        } else {
            return 4 * nodes[children[c]].eval(index, p, nodes);
        }
    }
};

class DTree {
public:
    const Vector3f& pos() const { return mPos; }

    float mean() const { return mMean; }

    float step() const {
        int dim = 1 << mDepth;
        return 1.0f / dim;
    }

    size_t nSamples() const {
        return mNumSamples;
    }

    bool read(BlobReader& blob) {
        uint64_t numNodes;
        uint64_t numSamples;
        blob >> mPos.x() >> mPos.y() >> mPos.z() >> mSize.x() >> mSize.y() >> mSize.z() >> mMean >> numSamples >> numNodes;
        if (!blob.isValid()) {
            return false;
        }

        mNumSamples = (size_t)numSamples;

        if (!isfinite(mMean)) {
            cerr << "INVALID MEAN: " << mMean << endl;
        }

        mPos += mSize / 2;

        mNodes.resize(numNodes);
        for (size_t i = 0; i < mNodes.size(); ++i) {
            auto& n = mNodes[i];
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < NUM_CHANNELS; ++k) {
                    blob >> n.data[k][j];
                    if (!isfinite(n.data[k][j])) {
                        cerr << "INVALID NODE: " << n.data[k][j] << endl;
                    }
                }
                blob >> n.children[j];
            }
        }

        mDepth = computeDepth();
        mMax = computeMax();

        return true;
    }

    float eval(int index, Vector2f p) const {
        if (mNumSamples == 0) {
            return 0;
        }
        const float factor = 1 / (float)(M_PI * mNumSamples);
        return factor * mNodes[0].eval(index, p, mNodes);
    }

    float evalIndex(int i, const Vector2i& index) const {
        const float localStep = step();
        const float offset = localStep * 0.5f;
        return mean() > 0 ? (eval(i, {1.0f - (index.y() * localStep + offset), index.x() * localStep + offset}) / mean()) : mean();
    }

    int loadData(vector<float>& data, int index) const {
        int dim = 1 << mDepth;
        float step = 1.0f / dim;
        float offset = step * 0.5f;

        float normFactor = mean() > 0 ? (1 / mean()) : mean();

        data.resize(dim * dim);

#pragma omp parallel for
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                data[i * dim + j] = eval(index, {1.0f - (offset + step * i), offset + step * j}) * normFactor;
            }
        }

        return dim;
    }

private:
    int computeDepth() const {
        return mNodes[0].computeDepth(mNodes) + 1;
    }

    float computeMax() const {
        if (mNumSamples == 0) {
            return 0;
        }
        const float factor = 1 / (float)(4 * M_PI * mNumSamples);
        return factor * mNodes[0].computeMax(mNodes);
    }

    int depth() const { return mDepth; }

    Vector3f mPos;
    Vector3f mSize;

    vector<QuadTreeNode> mNodes;
    float mMean;
    size_t mNumSamples;

    int mDepth;
    float mMax;
};

class GLTexture {
public:
    GLTexture() = default;
    GLTexture(const string& textureName)
        : mTextureName(textureName), mTextureId(0) {
    }

    GLTexture(GLTexture&& other)
        : mTextureName(move(other.mTextureName)),
        mTextureId(other.mTextureId) {
        other.mTextureId = 0;
    }

    GLTexture& operator=(GLTexture&& other) {
        mTextureName = move(other.mTextureName);
        swap(mTextureId, other.mTextureId);
        return *this;
    }

    ~GLTexture() {
        if (mTextureId)
            glDeleteTextures(1, &mTextureId);
    }

    GLuint texture(const DTree& dTree, int index) {
        load(dTree, index);
        return mTextureId;
    }

    const string& textureName() const { return mTextureName; }

private:
    void load(const DTree& dTree, int index) {
        if (mTextureId) {
            glDeleteTextures(1, &mTextureId);
            mTextureId = 0;
        }

        vector<float> data;
        int dim = dTree.loadData(data, index);

        glGenTextures(1, &mTextureId);
        glBindTexture(GL_TEXTURE_2D, mTextureId);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, dim, dim, 0, GL_RED, GL_FLOAT, &data[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    string mTextureName;
    GLuint mTextureId;
};

struct STree {
    array<ImageView*, NUM_CHANNELS> imageViews;
    array<GLTexture, NUM_CHANNELS> textures;
    vector<shared_ptr<DTree>> dTrees;

    Vector3f target;
    Vector3f eye;
    Vector3f up;

    size_t currentDTreeIndex = -1;

    float aabbDiag;
};

inline char separator() {
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

class SDTreeVisualizer : public Screen {
public:
    SDTreeVisualizer() : Screen(Vector2i(1024, 768), "SD-Tree Visualizer") {
        glfwMaximizeWindow(mGLFWWindow);

        Window* window = new Window(this, "Controls");
        window->setPosition(Vector2i(15, 15));
        window->setLayout(new GroupLayout());

        mImageContainer = new Window(this, "Samplingpoint dir");
        mImageContainer->setPosition(Vector2i(mSize.x() - 256, 0));
        mImageContainer->setLayout(new BoxLayout(Orientation::Vertical));
        mImageContainer->setFixedWidth(256);

        new Label(window, "Loading distributions", "sans-bold");
        Widget* tools = new Widget(window);
        tools->setLayout(new BoxLayout(Orientation::Horizontal,
            Alignment::Middle, 0, 6));

        Button* b = new Button(tools, "Open", ENTYPO_ICON_FOLDER);
        b->setCallback([&] {
            string file = file_dialog({{"sdt", "Sampling distribution"}}, false);
            if (!file.empty()) {
                loadSDTree(file);
            }
        });

        new Label(window, "Exposure", "sans-bold");
        Widget *panel = new Widget(window);
        panel->setLayout(new BoxLayout(Orientation::Horizontal,
            Alignment::Middle, 0, 20));

        mExposureSlider = new Slider(panel);
        mExposureSlider->setRange({-15.0f, 0.0f});
        mExposureSlider->setValue(-9.0f);
        mExposureSlider->setFixedWidth(120);

        mExposureTextbox = new TextBox(panel);
        mExposureTextbox->setFixedSize(Vector2i(60, 25));
        mExposureSlider->setCallback([this](float value) {
            setExposure(value);
        });
        mExposureTextbox->setFixedSize(Vector2i(60, 25));
        mExposureTextbox->setFontSize(20);
        mExposureTextbox->setAlignment(TextBox::Alignment::Right);

        setExposure(-9);

        mShader.init(
            "point_shader",

            /* Vertex shader */
            "#version 330\n"
            "uniform mat4 modelViewProj;\n"
            "in vec3 position;\n"
            "in vec3 attrColor;\n"
            "out vec3 fragColor;\n"
            "void main() {\n"
            "    gl_Position = modelViewProj * vec4(position, 1.0);\n"
            "    fragColor = attrColor;"
            "}",

            /* Fragment shader */
            "#version 330\n"
            "out vec4 color;\n"
            "in vec3 fragColor;\n"
            "void main() {\n"
            "    color = vec4(fragColor, 1.0);\n"
            "}"
        );

        performLayout();
    }

    ~SDTreeVisualizer() {
        mShader.free();
    }

    size_t totalPoints() const {
        size_t totalPoints = 0;
        for (const auto& sTree : mSDTrees) {
            totalPoints += sTree.dTrees.size();
        }
        return totalPoints;
    }

    void updateShader() {
        size_t nPoints = totalPoints();

        // Draw our dTrees
        MatrixXu indices(1, nPoints);
        MatrixXf positions(3, nPoints);

        size_t idx = 0;
        for (const auto& sTree : mSDTrees) {
            for (size_t i = 0; i < sTree.dTrees.size(); ++i) {
                const auto& dTree = *sTree.dTrees[i];
                indices.col(idx) << static_cast<unsigned int>(idx);
                positions.col(idx) = dTree.pos();
                ++idx;
            }
        }

        mShader.bind();
        mShader.uploadIndices(indices);
        mShader.uploadAttrib("position", positions);
    }

    float exposure() const {
        return mExposureSlider->value();
    }

    void setExposure(float exposure) {
        mExposureSlider->setValue(exposure);
        for (auto& sTree : mSDTrees) {
            for (auto& v : sTree.imageViews) {
                v->setExposure(exposure);
            }
        }

        stringstream stream;
        stream << fixed << setprecision(2) << exposure;
        mExposureTextbox->setValue(stream.str());
    }

    void loadSDTree(string filename) {
        size_t last = filename.find_last_of(separator()) + 1;
        new Label(mImageContainer, filename.substr(last, filename.size() - last), "sans-bold");

        BlobReader reader(filename);

        Matrix4f camera;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                reader >> camera(i, j);
            }
        }

        if (mSDTrees.size() == 0) {
            mCamera = camera.inverse();
            mCamera.row(0) *= -1;
            mCamera.row(2) *= -1;
            mUp = (mCamera.inverse().topLeftCorner(3, 3) * Vector3f(0, 1, 0)).normalized();
        }

        mSDTrees.emplace_back();
        STree& sTree = mSDTrees.back();

        sTree.dTrees.clear();

        Vector3f min = Vector3f::Constant(numeric_limits<float>::infinity());
        Vector3f max = Vector3f::Constant(-numeric_limits<float>::infinity());

        while (true) {
			shared_ptr<DTree> dTree = shared_ptr<DTree>(new DTree());
            if (!dTree->read(reader)) {
                break;
            }

            min = min.array().min(dTree->pos().array());
            max = max.array().max(dTree->pos().array());

            sTree.dTrees.emplace_back(move(dTree));
        }

        if (mSDTrees.size() == 1) {
            mAABBDiag = (max - min).norm();
        }

        cout << "Loaded SD-Tree with " << sTree.dTrees.size() << " D-Trees." << endl;

        const auto& firstDTree = *sTree.dTrees[0];

        Widget* buffers = new Widget(mImageContainer);
        buffers->setLayout(new BoxLayout(Orientation::Horizontal,
            Alignment::Middle, 0, 6));

        size_t idx = mSDTrees.size() - 1;
        for (size_t i = 0; i < sTree.imageViews.size(); ++i) {
            sTree.imageViews[i] = new ImageView(buffers, sTree.textures[i].texture(firstDTree, (int)i));

            sTree.imageViews[i]->setExposure(exposure());
            sTree.imageViews[i]->setGridThreshold(20);
            sTree.imageViews[i]->setPixelInfoThreshold(20);
            sTree.imageViews[i]->setPixelInfoCallback(
                [this, idx, i](const Vector2i& index) -> pair<string, Color> {
                    const STree& sTree = mSDTrees[idx];
                    auto& dTree = *sTree.dTrees[sTree.currentDTreeIndex];

                    const float radiance = dTree.evalIndex((int)i, index);

                    ostringstream oss;
                    oss << radiance << "\n";
                    if (dTree.mean() > 0) {
                        const float percent = 100 * dTree.step() * dTree.step() * radiance / dTree.mean();
                        oss << fixed;
                        oss.precision(2);
                        oss << percent << "%";
                    }

                    Color textColor = Color(0.0f, 1.0f);
                    return{oss.str(), textColor};
                }
            );
        }

        setDTree(sTree, 0);

        updateShader();
        updateColors();

        performLayout();
    }

    void updateColors() {
        size_t nPoints = totalPoints();
        MatrixXf colors(3, nPoints);

        size_t idx = 0;
        for (size_t j = 0; j < mSDTrees.size(); ++j) {
            const auto& sTree = mSDTrees[j];

            size_t cIdx = j + 1;
            Color c((float)(cIdx & 1), (float)((cIdx & 2) >> 1), (float)((cIdx & 4) >> 2), 1.0f);
            for (int i = 0; i < 3; ++i) {
                c[i] += 0.5f;
            }

            for (size_t i = 0; i < sTree.dTrees.size(); ++i) {
                if (i == sTree.currentDTreeIndex) {
                    colors.col(idx) << 1.0f, 1.0f, 1.0f;
                } else {
                    colors.col(idx) << c[0], c[1], c[2];
                }

                ++idx;
            }
        }

        mShader.bind();
        mShader.uploadAttrib("attrColor", colors);
    }

    Vector2f pixelToCanonical(const Vector2i& p) {
        return {(float)p.x() / size().x(), (float)p.y() / size().y()};
    }

    void setDTree(STree& sTree, size_t i) {
        if (sTree.currentDTreeIndex == i) {
            return;
        }

        sTree.currentDTreeIndex = i;
        DTree& dTree = *sTree.dTrees[i];

        Vector2i imageSize = Vector2i(256, 256);

        for (size_t j = 0; j < sTree.imageViews.size(); ++j) {
            sTree.imageViews[j]->bindImage(sTree.textures[j].texture(dTree, (int)j));
            sTree.imageViews[j]->setFixedSize(imageSize);
            sTree.imageViews[j]->setScale((float)imageSize.x() * dTree.step());
        }

        updateColors();
    }

    void updateDTree(STree& dist, const Vector2i &p) {
        // Obtain clicking point in [-1,1]^3 screen space.
        Vector2f point = (pixelToCanonical(p) * 2) - Vector2f{1, 1};
        Vector3f pEye = eye();

        // Map screen space point with some depth to scene point to obtain looking direction
        // of clicked location in world space.
        Vector4f dirHomo = mvp().inverse() * Vector4f(point.x(), -point.y(), 1, 1);
        Vector3f dir = (dirHomo.topRows(3) / dirHomo.w() - pEye).normalized();

        auto distPointDir = [](Vector3f o, Vector3f dir, Vector3f p) {
            Vector3f difference = p - o;
            float d = difference.norm();
            Vector3f toPoint = difference / d;
            float other = toPoint.dot(dir) * d;

            return sqrt(d * d - other * other);
        };

        vector<size_t> nearPoints;

        float eyeDist = dir.dot(pEye);

        // Find sampling point with shortest distance to ray parametrized by
        // o = mEye and d = dir
        float minDist = numeric_limits<float>::infinity();
        size_t minI = numeric_limits<size_t>::max();
        for (size_t i = 0; i < dist.dTrees.size(); ++i) {

            float distance = distPointDir(pEye, dir, dist.dTrees[i]->pos());

            if (distance < minDist) {
                minDist = distance;
                minI = i;
            }
        }

        if (minI == numeric_limits<size_t>::max()) {
            return;
        }

        // Visualize the selected sampling point
        setDTree(dist, minI);
    }

    bool dropEvent(const std::vector<std::string>& files) override {
        if (Screen::dropEvent(files)) {
            return true;
        }

        for (const auto& file : files) {
            loadSDTree(file);
        }

        return true;
    }

    bool mouseMotionEvent(const Vector2i &p, const Vector2i &rel, int button, int modifiers) override {
        if (Screen::mouseMotionEvent(p, rel, button, modifiers))
            return true;

        bool isRightHeld = (button & 2) != 0;
        if (isRightHeld) {
            mShallUpdatePosition = true;
            mPositionToUpdate = p;
        }

        Vector2f relF = pixelToCanonical(rel);
        Vector3f side = mUp.cross(eye()).normalized();

        bool isLeftHeld = (button & 1) != 0;
        if (isLeftHeld) {
            // Spin camera around target
            Matrix4f rot;
            rot.setIdentity();
            rot.topLeftCorner<3, 3>() = Matrix3f(
                AngleAxisf(static_cast<float>(relF.x() * 2 * M_PI), mUp) * // Scroll sideways around up vector
                AngleAxisf(static_cast<float>(relF.y() * 2 * M_PI), side) // Scroll around side vector
            );
            mCamera *= rot;
        }

        bool isMiddleHeld = (button & 4) != 0;
        if (isMiddleHeld) {
            // Translate camera
            Matrix4f trans;
            trans.setIdentity();
            trans.topRightCorner<3, 1>() += (-mUp * relF.y() + side * relF.x()) * mAABBDiag;
            mCamera *= trans;
        }

        return true;
    }

    bool mouseButtonEvent(const Vector2i &p, int button, bool down, int modifiers) override {
        if (Screen::mouseButtonEvent(p, button, down, modifiers))
            return true;

        bool isRightDown = down && button == 1;
        if (isRightDown) {
            mShallUpdatePosition = true;
            mPositionToUpdate = p;
        }

        return true;
    }

    bool scrollEvent(const Vector2i &p, const Vector2f &rel) override {
        if (Screen::scrollEvent(p, rel))
            return true;

        float scale = pow(1.1f, rel.y());
        mCamera.topRightCorner<3, 1>() /= scale;

        return true;
    }

    bool keyboardEvent(int key, int scancode, int action, int modifiers) override {
        if (Screen::keyboardEvent(key, scancode, action, modifiers))
            return true;
        if ((key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) && action == GLFW_PRESS) {
            setVisible(false);
            return true;
        }

        if (key == GLFW_KEY_E && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            if (modifiers & 1) {
                setExposure(exposure() - 0.5f);
            } else {
                setExposure(exposure() + 0.5f);
            }

            cout << "Exposure set to " << exposure() << endl;
            return true;
        }

        return false;
    }

    void syncImageViews() {
        // Sync up image views
        const auto& first = mSDTrees.front().imageViews.front();
        float firstStep = mSDTrees.front().dTrees[mSDTrees.front().currentDTreeIndex]->step();

        for (auto& sTree : mSDTrees) {
            float step = sTree.dTrees[sTree.currentDTreeIndex]->step();

            for (auto& v : sTree.imageViews) {
                if (v == first) {
                    continue;
                }

                v->setScale(step * first->scale() / firstStep);
                v->setOffset(first->offset());
            }
        }
    }

    void draw(NVGcontext *ctx) override {
        if (!mSDTrees.empty()) {
            syncImageViews();
        }

        if (mShallUpdatePosition) {
            for (auto& sTree : mSDTrees) {
                updateDTree(sTree, mPositionToUpdate);
            }

            mShallUpdatePosition = false;
        }

        Screen::draw(ctx);
    }

    Matrix4f proj() {
        Vector2i vSize = size();

        float fov = 50 * static_cast<float>(M_PI / 180);
        float aspect = (float)vSize.x() / vSize.y();
        float xScale = 1 / tan(fov / 2);
        float yScale = xScale * aspect;

        float zFar = 10000;
        float zNear = 0.01f;

        Matrix4f proj;
        proj << xScale, 0, 0, 0,
            0, yScale, 0, 0,
            0, 0, -(zFar + zNear) / (zFar - zNear), -1,
            0, 0, -2 * zNear*zFar / (zFar - zNear), 0;

        return proj.transpose();
    }

    Matrix4f mvp() {
        return proj() * mCamera;
    }

    Vector3f eye() {
        
        Vector4f eyeHomo = mCamera.inverse() * Vector4f(0, 0, 0, 1);
        return eyeHomo.topRows(3) / eyeHomo.w();
    }

    void drawContents() override {
        mShader.bind();
        mShader.setUniform("modelViewProj", mvp());
        mShader.drawIndexed(GL_POINTS, 0, static_cast<uint32_t>(totalPoints()));
    }

private:
    GLShader mShader;

    TextBox* mExposureTextbox;
    Slider* mExposureSlider;
    Widget* mImageContainer;

    vector<STree> mSDTrees;

    Matrix4f mCamera;
    Vector3f mUp;

    float mAABBDiag;

    Vector2i mPositionToUpdate;
    bool mShallUpdatePosition = false;
};

int main(int argc, char* argv[]) {
    try {
        init();

        {
            auto app = unique_ptr<SDTreeVisualizer>(new SDTreeVisualizer());
            app->setBackground(Color(0.0f, 0.0f, 0.0f, 1.0f));
            app->drawAll();
            app->setVisible(true);

            for (int i = 1; i < argc; ++i) {
                app->loadSDTree(argv[i]);
            }

            mainloop();
        }

        // glfwTerminate(), which is called by shutdown(), causes segmentation
        // faults on various linux distributions. Let's let the OS clean up
        // behind us and not crash in the last second.
        //shutdown();
    } catch (const runtime_error &e) {
        cerr << "Uncaught exception: " << e.what() << endl;
        return 1;
    }

    return 0;
}
