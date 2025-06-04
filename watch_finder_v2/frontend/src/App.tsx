import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { FiltersProvider } from "@/context/FiltersContext";
import Landing from "./pages/Landing";
import Index from "./pages/Index";
import LikedWatches from "./pages/LikedWatches";
import Filters from "./pages/Filters";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <FiltersProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/filters" element={<Filters />} />
            <Route path="/swipe" element={<Index />} />
            <Route path="/liked" element={<LikedWatches />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </FiltersProvider>
  </QueryClientProvider>
);

export default App;
