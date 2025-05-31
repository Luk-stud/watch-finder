/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ['extropian.b-cdn.net'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'extropian.b-cdn.net',
        pathname: '/data/brand_images/**',
      },
    ],
    // Allow local images in /public/images/
    unoptimized: false,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:5001/api/:path*',
      },
    ];
  },
};

module.exports = nextConfig; 