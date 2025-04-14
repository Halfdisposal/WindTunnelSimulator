// wind_tunnel_sfml.cpp
// Wind tunnel simulation ported from Python to C++ using SFML and Armadillo

#include <SFML/Graphics.hpp>
#include <armadillo>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>   // for std::max/std::min

// Grid parameters
const int N = 100;                // interior cells per side
const int WIN = 600;              // window size in pixels
const float CELL = WIN / float(N); // cell size in pixels

// Simulation parameters (initial values)
float dt = 0.1f;                  // timestep for advection & confinement
float damping = 0.99f;           // per-frame global damping
float strength = 200.0f;          // inflow strength (source injection)
float turbulence = 4.0f;         // vorticity confinement coefficient
float viscosity = 0.2f;         // diffusion coefficient
bool showPressure = false;        // toggle between velocity and pressure visualization
bool eraseMode = false;           // toggle for source erase

// Visualization colors
typedef sf::Color Color;
const Color minColor(31, 24, 38);
const Color maxColor(209, 155, 255);
const Color minPressColor(20, 20, 26);
const Color maxPressColor(175, 230, 255);
const float colorGamma = 0.5f;

// Fields (N+2 for ghost cells)
arma::mat u(N+2, N+2, arma::fill::zeros);
arma::mat v(N+2, N+2, arma::fill::zeros);
arma::mat u_new(N+2, N+2, arma::fill::zeros);
arma::mat v_new(N+2, N+2, arma::fill::zeros);
arma::mat pressure(N+2, N+2, arma::fill::zeros);
arma::mat pressure_new(N+2, N+2, arma::fill::zeros);

// Masks for obstacles and sources
std::vector<std::vector<bool>> obstacles(N+2, std::vector<bool>(N+2, false));
std::vector<std::vector<bool>> sources(N+2, std::vector<bool>(N+2, false));

// Convert grid indices to screen position
sf::Vector2f cellRect(int i, int j) {
    return sf::Vector2f((i-1) * CELL, (j-1) * CELL);
}

void diffuse() {
    for(int i = 1; i <= N; ++i) {
        for(int j = 1; j <= N; ++j) {
            if(obstacles[i][j]) {
                u_new(i,j) = 0.0f;
                v_new(i,j) = 0.0f;
            } else {
                u_new(i,j) = viscosity * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1));
                v_new(i,j) = viscosity * (v(i-1,j) + v(i+1,j) + v(i,j-1) + v(i,j+1));
            }
        }
    }
    u = u_new;
    v = v_new;
}

void advect(arma::mat &field, const arma::mat &u_vel, const arma::mat &v_vel) {
    arma::mat field0 = field;
    for(int i = 1; i <= N; ++i) {
        for(int j = 1; j <= N; ++j) {
            if(obstacles[i][j]) { field(i,j) = 0.0f; continue; }
            float x = i - dt * u_vel(i,j);
            float y = j - dt * v_vel(i,j);
            x = std::max(0.5f, std::min(float(N) + 0.5f, x));
            y = std::max(0.5f, std::min(float(N) + 0.5f, y));
            int i0 = int(std::floor(x));
            int j0 = int(std::floor(y));
            float s = x - i0;
            float t = y - j0;
            field(i,j) = (1-s)*(1-t)*field0(i0,j0)
                        + s*(1-t)*field0(i0+1,j0)
                        + (1-s)*t*field0(i0,j0+1)
                        + s*t*field0(i0+1,j0+1);
        }
    }
}

void addSource() {
    for(int i = 1; i <= N; ++i) {
        for(int j = 1; j <= N; ++j) {
            if(sources[i][j] && !obstacles[i][j]) {
                u(i,j) += strength;
            }
        }
    }
}

void vorticityConfinement() {
    arma::mat omega(N+2, N+2, arma::fill::zeros);
    for(int i=1; i<=N; ++i) for(int j=1; j<=N; ++j) {
        omega(i,j) = 0.5f * ((v(i+1,j) - v(i-1,j)) - (u(i,j+1) - u(i,j-1)));
    }
    for(int i=2; i< N; ++i) for(int j=2; j< N; ++j) {
        float dwdx = std::abs(omega(i+1,j)) - std::abs(omega(i-1,j));
        float dwdy = std::abs(omega(i,j+1)) - std::abs(omega(i,j-1));
        float mag = std::hypot(dwdx, dwdy) + 1e-5f;
        float Nx = dwdx / mag;
        float Ny = dwdy / mag;
        float fx = turbulence * Ny * omega(i,j);
        float fy = -turbulence * Nx * omega(i,j);
        if(!obstacles[i][j]) {
            u(i,j) += dt * fx;
            v(i,j) += dt * fy;
        }
    }
}

void applyObstacleFriction() {
    for(int i=1; i<=N; ++i) for(int j=1; j<=N; ++j) {
        if(obstacles[i][j]) {
            u(i,j) = 0.0f;
            v(i,j) = 0.0f;
        } else {
            if(obstacles[i-1][j] || obstacles[i+1][j]) u(i,j) = 0.0f;
            if(obstacles[i][j-1] || obstacles[i][j+1]) v(i,j) = 0.0f;
            u(i,j) *= damping;
            v(i,j) *= damping;
        }
    }
}

void computePressure() {
    arma::mat div(N+2, N+2, arma::fill::zeros);
    for(int i=1; i<=N; ++i) for(int j=1; j<=N; ++j) {
        if(obstacles[i][j]) continue;
        div(i,j) = -0.5f * ((u(i+1,j) - u(i-1,j)) + (v(i,j+1) - v(i,j-1)));
    }
    for(int it=0; it<40; ++it) {
        for(int i=1; i<=N; ++i) for(int j=1; j<=N; ++j) {
            if(obstacles[i][j]) continue;
            float s = div(i,j);
            for(auto &d : std::vector<std::pair<int,int>>{{1,0},{-1,0},{0,1},{0,-1}}) {
                int ni = i + d.first;
                int nj = j + d.second;
                if(obstacles[ni][nj]) s += pressure(i,j);
                else s += pressure(ni,nj);
            }
            pressure_new(i,j) = 0.25f * s;
        }
        pressure = pressure_new;
    }
}

void applyBoundaryConditions() {
    for(int j=1; j<=N; ++j) {
        if(!obstacles[1][j]) { u(0,j) = strength; v(0,j) = 0.0f; }
    }
    for(int j=1; j<=N; ++j) {
        u(N+1,j) = u(N,j);
        v(N+1,j) = v(N,j);
    }
    for(int i=0; i<=N+1; ++i) {
        u(i,0) = v(i,0) = u(i,N+1) = v(i,N+1) = 0.0f;
    }
}

void render(sf::RenderWindow &window) {
    float maxSpeed = 1e-5f;
    float minP = 1e5f, maxP = -1e5f;
    float minSpeed = 1e5f; // Initialize to a large value for comparison
    sf::Vector2f maxCoord, minCoord; // To store coordinates if needed

    for(int i = 1; i <= N; ++i) {
        for(int j = 1; j <= N; ++j) {
            if (!obstacles[i][j]) {
                float speed = std::hypot(u(i, j), v(i, j));
                maxSpeed = std::max(maxSpeed, speed);
                minSpeed = std::min(minSpeed, speed);
                // Optionally save the location of the min/max speeds
                if (speed == maxSpeed) maxCoord = sf::Vector2f(i, j);
                if (speed == minSpeed) minCoord = sf::Vector2f(i, j);
            }
        }
    }

    if (showPressure) {
        // Calculate min and max pressures
        for (int i = 1; i <= N; ++i) {
            for (int j = 1; j <= N; ++j) {
                if (!obstacles[i][j]) {
                    minP = std::min(minP, float(pressure(i, j)));
                    maxP = std::max(maxP, float(pressure(i, j)));
                }
            }
        }
        if (maxP <= minP) maxP = minP + 1.0f;
    } else {
        // Calculate min and max velocities
        for (int i = 1; i <= N; ++i) {
            for (int j = 1; j <= N; ++j) {
                if (!obstacles[i][j]) {
                    float speed = std::hypot(u(i, j), v(i, j));
                    maxSpeed = std::max(maxSpeed, speed);
                    minSpeed = std::min(minSpeed, speed);
                }
            }
        }
    }
    
        // Set the window title based on the mode
    if (showPressure) {
        window.setTitle("Pressure Mode: Max = " + std::to_string(maxP) +
                        ", Min = " + std::to_string(minP));
    } else {
        window.setTitle("Velocity Mode: Max = " + std::to_string(maxSpeed) +
                        ", Min = " + std::to_string(minSpeed));
    }
    
    
    if(showPressure) {
        for(int i=1;i<=N;++i) for(int j=1;j<=N;++j) if(!obstacles[i][j]) {
            minP = std::min(minP, float(pressure(i,j)));
            maxP = std::max(maxP, float(pressure(i,j)));
        }
        if(maxP <= minP) maxP = minP + 1.0f;
    } else {
        for(int i=1;i<=N;++i) for(int j=1;j<=N;++j) if(!obstacles[i][j]) {
            maxSpeed = std::max(maxSpeed, static_cast<float>(std::hypot(u(i,j), v(i,j))));
        }
    }


    sf::RectangleShape cell(sf::Vector2f(CELL+1, CELL+1));
    for(int i=1;i<=N;++i) {
        for(int j=1;j<=N;++j) {
            Color color;
            if(showPressure) {
                float t = (pressure(i,j) - minP) / (maxP - minP);
                int r = int(minPressColor.r + t*(maxPressColor.r - minPressColor.r));
                int g = int(minPressColor.g + t*(maxPressColor.g - minPressColor.g));
                int b = int(minPressColor.b + t*(maxPressColor.b - minPressColor.b));
                color = Color(r,g,b);
                if(obstacles[i][j]) {
                    color = Color(255,215,0);
                } else if(sources[i][j]) {
                    color = Color(0,255,0); 
                }
            } else if(obstacles[i][j]) {
                color = Color(255,215,0);
            } else if(sources[i][j]) {
                color = Color(0,255,0);
            } else {
                float speed = std::hypot(u(i,j), v(i,j));
                float t = std::pow(std::min(speed / maxSpeed, 1.0f), colorGamma);
                int r = int(minColor.r + t*(maxColor.r - minColor.r));
                int g = int(minColor.g + t*(maxColor.g - minColor.g));
                int b = int(minColor.b + t*(maxColor.b - minColor.b));
                color = Color(r,g,b);
            }
            cell.setFillColor(color);
            cell.setPosition((i-1)*CELL, (j-1)*CELL);
            window.draw(cell);
        }
    }
}

int main() {
    sf::RenderWindow window(sf::VideoMode(WIN, WIN), "Wind Tunnel (SFML + Armadillo)");
    window.setFramerateLimit(60);
    while(window.isOpen()) {
        sf::Event ev;
        while(window.pollEvent(ev)) {
            if(ev.type == sf::Event::Closed) window.close();
            else if(ev.type == sf::Event::KeyPressed) {
                switch(ev.key.code) {
                    case sf::Keyboard::P: showPressure = !showPressure; break;
                    case sf::Keyboard::E: eraseMode = !eraseMode; break;
                    case sf::Keyboard::Q: dt = std::max(0.01f, dt - 0.01f); break;
                    case sf::Keyboard::W: dt = std::min(1.0f, dt + 0.01f); break;
                    case sf::Keyboard::A: damping = std::max(0.0f, damping - 0.01f); break;
                    case sf::Keyboard::S: damping = std::min(1.0f, damping + 0.01f); break;
                    case sf::Keyboard::Z: strength = std::max(0.0f, strength - 1.0f); break;
                    case sf::Keyboard::X: strength = std::min(100.0f, strength + 1.0f); break;
                    case sf::Keyboard::R: turbulence = std::max(0.0f, turbulence - 0.5f); break;
                    case sf::Keyboard::T: turbulence = std::min(50.0f, turbulence + 0.5f); break;
                    case sf::Keyboard::F: viscosity = std::max(0.0f, viscosity - 0.01f); break;
                    case sf::Keyboard::G: viscosity = std::min(1.0f, viscosity + 0.01f); break;
                    default: break;
                }
            }
        }
        if(sf::Mouse::isButtonPressed(sf::Mouse::Left) ||
           sf::Mouse::isButtonPressed(sf::Mouse::Right) ||
           sf::Mouse::isButtonPressed(sf::Mouse::Middle)) {
            auto pos = sf::Mouse::getPosition(window);
            int ci = pos.x / CELL + 1;
            int cj = pos.y / CELL + 1;
            if(ci>=1 && ci<=N && cj>=1 && cj<=N) {
                if(sf::Mouse::isButtonPressed(sf::Mouse::Left)) obstacles[ci][cj] = true;
                else if(sf::Mouse::isButtonPressed(sf::Mouse::Right)) obstacles[ci][cj] = false;
                else if(sf::Mouse::isButtonPressed(sf::Mouse::Middle)) sources[ci][cj] = !eraseMode;
            }
        }
        addSource();
        diffuse();
        advect(u, u, v);
        advect(v, u, v);
        vorticityConfinement();
        applyObstacleFriction();
        double total_v = 0.0; int count = 0;
        for(int i=1;i<=N;++i) for(int j=1;j<=N;++j) if(!obstacles[i][j]) { total_v += v(i,j); ++count; }
        double avg_v = count>0 ? total_v/count : 0.0;
        for(int i=1;i<=N;++i) for(int j=1;j<=N;++j) if(!obstacles[i][j]) v(i,j) -= avg_v;
        if(showPressure) computePressure();
        applyBoundaryConditions();
        window.clear(sf::Color(30,30,30));
        render(window);
        window.display();
    }
    return 0;
}
