#ifndef MORPH_HPP_
#define MORPH_HPP_

# include <iostream>
# include <vector>
# include <algorithm>
# include <functional>
# include <deque>
# include <stack>
# include <opencv2/opencv.hpp>


// source:https://sites.google.com/site/alexzaitsevcodesamples/home/image-processing/extendedmax-reconstruct-regmax-c-implementation 
namespace morph {
    enum { LEADING = 1, TRAILING = 2, CENTER = 4 };
    struct PTS {
        int x = 0;
        int y = 0;
    };

    // width : cols, height: rows
    template <typename TYPE>
    void Reconstruct(TYPE* const Marker, const TYPE* const Mask, int  Width, int  Height)

    {

        Reconstruct(Marker, Mask, Width * Height, NeighborhoodWalker2D(Width, Height, LEADING | TRAILING),

            NeighborhoodWalker2D(Width, Height, TRAILING),

            NeighborhoodWalker2D(Width, Height, LEADING));

    }



    template <typename TYPE>

    void Regmax(const TYPE* const F, bool* const BW, int Width, int Height)

    {

        Regmax(F, BW, Width * Height, NeighborhoodWalker2D(Width, Height, LEADING | TRAILING));

    }



    template <typename TYPE>

    void ExtendedMax(const TYPE* const F, bool* const BW, const TYPE Gap, int Width, int Height)

    {

        const int N(Width * Height);

        std::vector<TYPE> Marker; Marker.reserve(N);

        std::transform(F, F + N, std::back_inserter(Marker), std::bind2nd< std::minus<TYPE>, TYPE >(std::minus<TYPE>(), Gap));

        Reconstruct(&Marker[0], F, Width, Height);

        Regmax(&Marker[0], BW, Width, Height);

    }



    template <typename TYPE, class NeighborhoodWalker>

    void Regmax(const TYPE* const F, bool* const BW, int N, NeighborhoodWalker& walker)

    {

        std::stack<int> stack;

        std::fill_n(BW, N, true);

        for (int p(0); p < N; ++p) {

            if (!BW[p])   continue;

            walker.SetLocation(p);

            bool found(false);

            for (int q(0); !found && walker.GetNextInboundsNeighbor(q); found = F[q] > F[p]) {}

            if (!found)  continue;

            const TYPE val(F[p]);

            stack.push(p);

            BW[p] = 0;

            while (!stack.empty()) {

                const int pp(stack.top()); stack.pop();

                walker.SetLocation(pp);

                for (int qq(0); walker.GetNextInboundsNeighbor(qq); ) {

                    if (BW[qq] && F[qq] == val) {

                        stack.push(qq);

                        BW[qq] = 0;

                    }

                }

            }

        }

    }


    template <typename TYPE, class NeighborhoodWalker>

    int Reconstruct(TYPE* const J, const TYPE* const I, int N,

        NeighborhoodWalker& walker,

        NeighborhoodWalker& trailing_walker,

        NeighborhoodWalker& leading_walker)

    {

        struct EQ { bool operator()(const TYPE& i, const TYPE& j) { return i >= j; } };

        if (!std::equal(I, I + N, J, EQ()))

            return -1;

        std::deque<int> queue;

        for (int p(0); p < N; ++p) {

            int maxpixel(J[p]);

            trailing_walker.SetLocation(p);

            for (int q(0); trailing_walker.GetNextInboundsNeighbor(q); ) {

                if (J[q] > maxpixel) {

                    maxpixel = J[q];

                }

            }

            J[p] = maxpixel < I[p] ? maxpixel : I[p];

        }



        for (int p(N - 1); p >= 0; --p) {

            int maxpixel(J[p]);

            leading_walker.SetLocation(p);

            for (int q(0); leading_walker.GetNextInboundsNeighbor(q); ) {

                if (J[q] > maxpixel) {

                    maxpixel = J[q];

                }

            }

            J[p] = maxpixel < I[p] ? maxpixel : I[p];

            leading_walker.SetLocation(p);

            for (int q(0); leading_walker.GetNextInboundsNeighbor(q); ) {

                if (J[q] < J[p] && J[q] < I[q]) {

                    queue.push_back(p);

                    break;

                }

            }

        }



        while (!queue.empty()) {

            const int p(queue.front());         queue.pop_front();

            const int Jp(J[p]);

            walker.SetLocation(p);

            for (int q(0); walker.GetNextInboundsNeighbor(q); ) {

                const int Jq(J[q]), Iq(I[q]);

                if (Jq < Jp && Iq != Jq) {

                    J[q] = Jp < Iq ? Jp : Iq;

                    queue.push_back(q);

                }

            }

        }

        return 0;

    }



    
    class NeighborhoodWalker2D

    {

        const int width_, height_;

        PTS points_[9], * ipoint_, * points_end_, pos_;

        inline int sub2ind(int x, int y) { return y * width_ + x; }

        NeighborhoodWalker2D(const NeighborhoodWalker2D&);

    public:

        inline void SetLocation(int p)

        {

            pos_.x = p % width_;

            pos_.y = p / width_;

            ipoint_ = points_;

        }

        bool GetNextInboundsNeighbor(int& q)

        {

            for (; ipoint_ != points_end_; ) {

                const PTS p = { ipoint_->x + pos_.x, ipoint_->y + pos_.y };

                ++ipoint_;

                if (0 <= p.x && p.x < width_ && 0 <= p.y && p.y < height_) {

                    q = sub2ind(p.x, p.y);

                    return true;

                }

            }

            return false;

        }

        NeighborhoodWalker2D(int width, int height, int flag) :ipoint_(points_), width_(width), height_(height)

        {

            const bool center(CENTER & flag), leading(LEADING & flag), trailing(TRAILING & flag);

            for (int y(-1); y <= 1; ++y) {

                for (int x(-1); x <= 1; ++x) {

                    const int d(sub2ind(x, y));

                    if ((center && !d) || (leading && d < 0) || (trailing && d > 0)) {

                        ipoint_->x = x;

                        ipoint_->y = y;

                        ++ipoint_;

                    }

                }

            }
            points_end_ = ipoint_;
        }
    };


    std::vector<cv::Mat>  CaculateMP(const cv::Mat& src, int morph_size =3);

    // data type requried by imreconstruct.cpp
    auto* matToArray(const cv::Mat& image);

    cv::Mat imReconstruct(const cv::Mat& marker, const cv::Mat& mask);
    cv::Mat imRegionalMax(const cv::Mat& src);
}



#endif
