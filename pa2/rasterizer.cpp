// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool onTheLine(Vector3f a, Vector3f b, Vector3f q)
{
    auto ab = b-a;
    auto aq = q-a;
    if(ab.cross(aq)!= Vector3f(0,0,0))
        return false;

    auto minX = MIN(a.x(),b.x());
    auto maxX = MAX(a.x(),b.x());
    
    auto minY = MIN(a.x(),b.x());
    auto maxY = MAX(a.y(),b.y());

    if(q.x()< minX)
    return false;

    if(q.x()> maxX)
    return false;

    if(q.y()< minY)
    return false;

    if(q.y()> maxY)
    return false;

    return true;
}

static Vector3f  insideTriangle_helper(Vector3f a, Vector3f b, Vector3f q)
{
    auto ab = b-a;
    auto aq = q-a;
    auto norm = ab.cross(aq);

    //zero-vector keep unchanged after normalizing
    norm.normalize();
    return norm;
}

static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    //Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]

    Vector3f q = Vector3f(x,y,0);
    Vector3f p0 = Vector3f(_v[0].x(),_v[0].y(),0);
    Vector3f p1 = Vector3f(_v[1].x(),_v[1].y(),0);
    Vector3f p2 = Vector3f(_v[2].x(),_v[2].y(),0);

    if(onTheLine(p0,p1,q))
        return true;
    if(onTheLine(p1,p2,q))
        return true;
    if(onTheLine(p2,p0,q))
        return true;

    auto v0 = insideTriangle_helper(p0,p1,q);
    auto v1 = insideTriangle_helper(p1,p2,q);
    auto v2 = insideTriangle_helper(p2,p0,q);

    if(v0!=v1)
        return false;
    if(v1!=v2)
        return false;

    return true;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (100 - 0.1) / 2.0;
    float f2 = (100 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    //Find out the bounding box of current triangle.
    float minX,maxX,minY,maxY;
    minX = MIN(t.v[0].x(),t.v[1].x());
    minX = MIN(minX,t.v[2].x());
    minX = floor(minX);

    maxX = MAX(t.v[0].x(),t.v[1].x());
    maxX = MAX(maxX,t.v[2].x());
    maxX = ceil(maxX);

    minY = MIN(t.v[0].y(),t.v[1].y());
    minY = MIN(minY,t.v[2].y());
    minY = floor(minY);

    maxY = MAX(t.v[0].y(),t.v[1].y());
    maxY = MAX(maxY,t.v[2].y());
    maxY = ceil(maxY);

    // iterate through the pixel and find if the current pixel is inside the triangle
    for (int i = minX; i < maxX; i++)
    {
        for (int j = minY; j < maxY; j++)
        {
            float depth = 0;
            Vector3f color = Vector3f(0,0,0);
            std::vector<Vector2f> children = {Vector2f(i+0.25f,j+0.75f),Vector2f(i+0.25f,j+0.25f),Vector2f(i+0.75f,j+0.25f),Vector2f(i+0.75f,j+0.75f)};

            bool inside = false;
            for(auto &child : children)
            {
                if(insideTriangle(child.x(),child.y(),t.v))
                {
                    auto[alpha, beta, gamma] = computeBarycentric2D(child.x(), child.y(), t.v);
                    float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;

                    depth = z_interpolated;
                    color += t.color[0] * 0.25f;
                    inside = true;
                }
            }

            if(!inside)
                continue;

            auto index = get_index(i,j);
            if( depth_buf[index] > depth)
            {
                depth_buf[index] = depth;
                set_pixel(Vector3f(i,j,0),255*color);
            }
            
            // if(insideTriangle(i+0.5f, j+0.5f, t.v))
            // {
                
            //     //get the depth of(i+0.5f, j+0.5f)
            //     //If so, use the following code to get the interpolated z value.
            //     auto[alpha, beta, gamma] = computeBarycentric2D(i+0.5f, j+0.5f, t.v);
            //     float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            //     float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
            //     z_interpolated *= w_reciprocal;

            //     auto index = get_index(i,j);
            //     if( depth_buf[index] > z_interpolated)
            //     {
            //         depth_buf[index] = z_interpolated;
            //         set_pixel(Vector3f(i,j,0),255*t.color[0]);
            //     }
            // }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    auto ind = get_index(point.x(),point.y());
    frame_buf[ind] = color;
}