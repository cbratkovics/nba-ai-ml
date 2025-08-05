/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['recharts'],
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      'react-is': require.resolve('react-is'),
    };
    return config;
  },
}

module.exports = nextConfig