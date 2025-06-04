/**
 * Connection Test Utility
 * Test multiple API endpoints to diagnose connection issues
 */

export interface ConnectionTestResult {
  url: string;
  success: boolean;
  responseTime: number;
  error?: string;
  status?: number;
}

export async function testConnection(url: string, timeout: number = 5000): Promise<ConnectionTestResult> {
  const startTime = Date.now();
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    const responseTime = Date.now() - startTime;
    
    return {
      url,
      success: response.ok,
      responseTime,
      status: response.status,
      error: response.ok ? undefined : `HTTP ${response.status}: ${response.statusText}`,
    };
  } catch (error) {
    const responseTime = Date.now() - startTime;
    return {
      url,
      success: false,
      responseTime,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

export async function testMultipleConnections(urls: string[]): Promise<ConnectionTestResult[]> {
  console.log(`🔍 Testing ${urls.length} API endpoints...`);
  
  const results = await Promise.all(
    urls.map(url => testConnection(`${url}/health`))
  );
  
  // Log results
  results.forEach((result, index) => {
    const emoji = result.success ? '✅' : '❌';
    const time = result.responseTime;
    console.log(`${emoji} ${result.url} - ${time}ms ${result.error ? `(${result.error})` : ''}`);
  });
  
  const successfulUrls = results.filter(r => r.success);
  if (successfulUrls.length > 0) {
    console.log(`🎉 Found ${successfulUrls.length} working API endpoint(s)`);
    console.log('Fastest endpoint:', successfulUrls.sort((a, b) => a.responseTime - b.responseTime)[0].url);
  } else {
    console.error('🚨 No working API endpoints found');
  }
  
  return results;
}

export async function findWorkingApiUrl(candidateUrls: string[]): Promise<string | null> {
  const results = await testMultipleConnections(candidateUrls);
  const working = results.filter(r => r.success);
  
  if (working.length === 0) {
    return null;
  }
  
  // Return the fastest working URL
  const fastest = working.sort((a, b) => a.responseTime - b.responseTime)[0];
  return fastest.url.replace('/health', '');
} 