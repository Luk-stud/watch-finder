import React from 'react';
import { Link } from 'react-router-dom';
import { ChevronRight, Settings, Heart, Brain, Sparkles, Filter } from 'lucide-react';
import { useViewportHeight } from '../hooks/useViewportHeight';

const Landing = () => {
  // Use the viewport height hook for mobile browser compatibility
  useViewportHeight();

  return (
    <div className="flex-viewport bg-background">
      {/* Header */}
      <header className="flex-shrink-0 p-6 md:p-8">
        <div className="flex items-center justify-center gap-3">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <Settings className="w-4 h-4 text-primary-foreground" />
          </div>
          <h1 className="text-xl md:text-2xl font-light tracking-[0.2em] text-foreground uppercase">
            WatchSwipe AI
        </h1>
      </div>
      </header>

      {/* Main Content - Scrollable if needed on very small screens */}
      <main className="flex-1 flex flex-col items-center justify-center px-6 md:px-8 text-center max-w-2xl mx-auto min-h-0 overflow-y-auto">
        {/* Hero Section */}
        <div className="mb-8 md:mb-12">
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-light text-foreground mb-4 md:mb-6 leading-tight tracking-wide">
            Discover
            <span className="block text-muted-foreground font-normal">Smart Timepieces</span>
          </h2>
          
          <p className="text-base md:text-lg text-muted-foreground leading-relaxed font-light max-w-md mx-auto tracking-wide">
            AI-powered watch discovery that learns your preferences. 
            Each swipe reveals craftsmanship tailored to your taste.
          </p>
        </div>

        {/* Features */}
        <div className="grid grid-cols-1 gap-6 mb-8 md:mb-12 max-w-sm">
          <div className="text-center">
            <div className="w-12 h-12 bg-primary/10 border border-primary/20 rounded-full flex items-center justify-center mx-auto mb-3">
              <Brain className="w-5 h-5 text-primary" />
            </div>
            <h3 className="text-foreground font-medium mb-1 text-sm tracking-wide">AI Learning</h3>
            <p className="text-muted-foreground text-xs font-light tracking-wide leading-relaxed">
                              Advanced recommendation algorithm adapts to your preferences with every swipe
            </p>
          </div>
          
          <div className="text-center">
            <div className="w-12 h-12 bg-green-500/10 border border-green-500/20 rounded-full flex items-center justify-center mx-auto mb-3">
              <Heart className="w-5 h-5 text-green-600 dark:text-green-400" />
            </div>
            <h3 className="text-foreground font-medium mb-1 text-sm tracking-wide">Smart Recommendations</h3>
            <p className="text-muted-foreground text-xs font-light tracking-wide leading-relaxed">
              Personalized suggestions that improve as you interact with the app
            </p>
          </div>
          
          <div className="text-center">
            <div className="w-12 h-12 bg-accent border border-border rounded-full flex items-center justify-center mx-auto mb-3">
              <Sparkles className="w-5 h-5 text-accent-foreground" />
            </div>
            <h3 className="text-foreground font-medium mb-1 text-sm tracking-wide">Detailed Exploration</h3>
            <p className="text-muted-foreground text-xs font-light tracking-wide leading-relaxed">
              Explore complete specifications and build your personal collection
            </p>
          </div>
        </div>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 items-center">
          <Link
            to="/filters"
            className="group inline-flex items-center gap-3 bg-secondary hover:bg-secondary/80 text-secondary-foreground px-6 md:px-8 py-3 md:py-4 text-base md:text-lg font-medium rounded-md transition-all duration-300 hover:shadow-lg tracking-wide focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          >
            <Filter className="w-4 h-4" />
            Customize Preferences
          </Link>
          
        <Link
          to="/swipe"
            className="group inline-flex items-center gap-3 bg-primary hover:bg-primary/90 text-primary-foreground px-6 md:px-8 py-3 md:py-4 text-base md:text-lg font-medium rounded-md transition-all duration-300 hover:shadow-lg tracking-wide focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
        >
            Quick Start
            <ChevronRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
        </Link>
        </div>
        
        <p className="text-muted-foreground text-xs mt-4 font-light tracking-wide">
          Start your intelligent horological journey
        </p>
      </main>

      {/* Footer */}
      <footer className="flex-shrink-0 p-6 md:p-8 text-center">
        <div className="w-16 h-px bg-border mx-auto"></div>
        <p className="text-xs text-muted-foreground mt-4 font-light tracking-wide">
          Powered by advanced machine learning algorithms
        </p>
      </footer>
    </div>
  );
};

export default Landing;
