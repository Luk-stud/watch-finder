/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true,
    domains: ['extropian.b-cdn.net'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'extropian.b-cdn.net',
        pathname: '/data/brand_images/**',
      },
    ],
  },
  // Remove rewrites for static export
};

module.exports = nextConfig; 